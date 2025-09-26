import os
import copy
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from monai.transforms import Compose
from skimage.morphology import skeletonize
import networkx.algorithms.approximation as nx_approx

from src.utils.geometry import transform_points
from src.tester.tracker_inference_base import TrackerInferenceBase
from src.tester.adasire_transforms import (
    IcoSphereLocalMaxSolver,
    AdaSireSampleAndPredict,
    AdaSireScalesNormMax,
    AdaSirePredictionNormalize,
    AdaSirePhyRadius,
)
from src.utils.image_transforms import InferImgPreTransform
from src.utils.graph_transforms import (
    BuildMultiScaleGraphStructure,
    FillSamplingCoords,
    BuildForGEMGCN,
    GetKHopUndirectedEdges,
)
from src.tester.tracked_centerline import ANCHOR
from src.tester.stopping_criterions import (
    MaxIterationsStoppingCriterion,
    AlreadyTrackedStoppingCriterion,
    OutofSegmentStoppingCriterion,
    MaxProbStoppingCriterion,
)


class WaveFrontTrackingPipeline(TrackerInferenceBase):
    """
    Tracking pipeline that propagates in different directions like waves.
    Serve as base class for different direction & radius estimators. In this
    class we only implement the core tracking logic. Image and seeds management
    implementations are left to subclasses.
    Args:
        config(BaseConfig): A config object describing tracking configuration.
        log_dir(str): A file location to store tracking logs and results.
    """

    def __init__(self, config, log_dir):

        self.explore_fronts: List[ANCHOR] = []
        # Initialize base class attributes
        super().__init__(config, log_dir)

    def _build_stopping_criteria(self, **kwds):
        """
        Add method-specific stopping criteria for tracking.
        """

        super()._build_stopping_criteria()  # Default stopping criterion
        maxIters = MaxIterationsStoppingCriterion(
            max_iterations=kwds.get("MaximumIterations", 600))
        tracked = AlreadyTrackedStoppingCriterion(
            None, distance=kwds.get("TrackedTrajectoryMaximumDistance", 0.1))
        self.stop_criteria.update(
            {
                "MaxIteration": maxIters,
                "Tracked": tracked,
            }
        )

    def add_to_front(
        self,
        pos: torch.Tensor | np.ndarray | List,
        node: int,
        radius: float,
    ):
        """ Update the anchor points list. """

        if isinstance(pos, List):
            self.logger.info(f"Queuing exploration front at {pos}.")
            self.explore_fronts.append(
                ANCHOR(node=node, position=pos, radius=radius))
        elif isinstance(pos, (np.ndarray, torch.Tensor)):
            self.logger.info(f"Queuing exploration front at {pos.tolist()}.")
            self.explore_fronts.append(
                ANCHOR(node=node, position=pos.tolist(), radius=radius))
        else:
            raise ValueError(f"Type of pos {type(pos)} invalid.")

    def filter_point(
        self,
        query: torch.Tensor,
        exclude_node: int | None,
    ) -> Tuple[bool, List[int]]:
        """
        Filter out invalid anchor positions. Return True if query is valid,
        that is, query point is not lying inside radial ball of any current
        centerline points.
        """

        if exclude_node is None:
            nodes = [n for n in self.centerline.centerline_graph_dict["nodes"]]
        else:
            nodes = [n for n in self.centerline.centerline_graph_dict["nodes"]
                     if n["id"] != exclude_node]
        if not nodes:
            return True, [-1]

        dev = query.device
        pos = torch.tensor([n["attributes"]["pos"] for n in nodes]).to(dev)
        r = torch.tensor([n["attributes"]["r"] for n in nodes]).to(dev)

        dist = torch.linalg.norm(query.view(-1, 3) - pos.view(-1, 3), dim=-1)
        valid = torch.all(dist >= r).item()
        if valid:
            return True, [-1]

        invalid_inds = torch.argwhere(dist < r).squeeze(-1).tolist()
        close_nodes = [nodes[ind]["id"] for ind in invalid_inds]
        return valid, close_nodes

    def add_connection(
        self,
        node: int,
        pos: torch.Tensor,
        radius: float,
        candidates: List[int],
    ):
        """
        Add (possible) connections between wave-fronts.
        """

        for t in candidates:
            conn = nx_approx.node_connectivity(
                self.centerline.centerline_graph_nx, s=node, t=t)
            if conn == 0:  # Disconnected
                new_node = self.update_centerline(pos, node, r=radius)
                self.centerline.add_edge(new_node, t)
                self.logger.warning(f"Adding edge between {(new_node, t)}.")

    def update_front(self, data: Dict[str, Any], iteration: int, **kwds):
        """ Expand current exploration fronts. Substitute exploration fronts
        with new anchors. """

        fronts = copy.deepcopy(self.explore_fronts)
        self.explore_fronts.clear()  # Prepare for new anchor points
        for anchor in fronts:
            point, node, r = anchor.position, anchor.node, anchor.radius
            point = torch.tensor(point).to(self.device)

            self.logger.info(f"AT node {node}: pos {point} radius {r}.")

            # Direction and radius estimation
            data.update({
                "point": point.clone(),
                "normalization": kwds["normalization"],
            })
            dir_meta = self.direction_transform(data)
            directions = dir_meta["directions"]
            # pred = dir_meta["pred"]

            # Update current centerline graph node with new attributes
            self.centerline.update_node_attr(node, dirs=directions)

            # Signal warning message if no direction is found.
            if not directions:
                self.logger.warning("No valid dir_scales acquired.")
                continue

            # Validate and update next-step wave front points
            for direction in directions:
                pos, _ = self.move_one_step(point, torch.tensor(direction), r)
                # Check stopping criteria at this possibly new anchor point.
                self.logger.info(
                    f"Validating direction {direction} with step size {r} "
                    f"bringing to new position {pos.tolist()} ...")

                stopping_states = self.check_stopping_criterions(
                    point=pos.clone(),  # Out of bound or close to tracked
                    # probabilities=pred,  # Probability-related stopping
                )
                if np.any(list(stopping_states.values())):
                    self._get_triggered_criterion(stopping_states, iteration)
                    continue
                # Infer radius at this position.
                data.update({"point": pos, })
                dir_meta = self.direction_transform(data)
                new_r = dir_meta["radius"]
                # Check if the new node falls near tracked centerline
                valid_front, close_nodes = self.filter_point(pos, node)
                if not valid_front:
                    self.logger.info(f"{pos.tolist()} near {close_nodes}.")
                    self.add_connection(node, pos, new_r, close_nodes)
                    continue

                self.logger.info(f"{pos.tolist()} is valid front.")
                new_node = self.update_centerline(pos, node, r=new_r)
                self.add_to_front(pos, new_node, new_r)
            # Update stopping criterion attributes
            self.update_stopping(
                "Tracked", new_points=self.centerline.centerline_points)
        self.logger.warning(f"{len(self.explore_fronts)} fronts maintained.")

    def track_from_seed(
        self,
        data: Dict[str, Any],
        seed_point: torch.Tensor,
        **kwds,
    ) -> None:
        """
        Track from one seed using wave-front propagation.
        """

        self.explore_fronts.clear()

        # Infer radius at this location
        data.update({
            "point": seed_point,
            "normalization": kwds["normalization"],
        })
        dir_meta = self.direction_transform(data)
        r = dir_meta["radius"]
        # Enque seed in the exploration front and update centerline.
        node = self.update_centerline(seed_point, -1, r=r)
        self.add_to_front(seed_point, node, r)

        # Propagate wave fronts
        iteration = 0
        while self.explore_fronts:
            iteration += 1
            if len(self.explore_fronts) > kwds["max_front_len"]:
                self.logger.warning("Wave front length exceeds maximum.")
                break
            self.update_front(data, iteration, **kwds)
            stopping_criterion_states = self.check_stopping_criterions(
                iteration=iteration,  # Maximum iteration
            )
            if np.any(list(stopping_criterion_states.values())):
                break

        self._get_triggered_criterion(stopping_criterion_states, iteration)


class WaveFrontFixSeeds(WaveFrontTrackingPipeline):
    """
    Adapt the wave-front propagation tracking pipeline to fixed-seeds init
    scenario. Only run_tracking() and update_seeds() function is changed.
    """

    def update_seeds(self, seeds: List, **kwds):

        current_track = self.centerline.centerline_points.to(self.device)
        # During initialization, all skeleton points are added to the queue.
        if len(current_track) == 0:  # Init condition.
            self.seeds_pool = copy.deepcopy(seeds)
        else:
            pass  # No operations

        self.logger.warning("%d SEEDs Remaining." % len(self.seeds_pool))

    def run_tracking(
        self,
        case_id: str,
        image_name: str,
        **kwds,
    ) -> Dict[List[int | float], Any]:
        """ Run segmentator inference on one image. """

        image, affine, spacing = self.init_on_new_image(image_name, **kwds)
        self.update_seeds(seeds=kwds["seeds"])

        # Get pre-transformed data
        data = {
            "id": case_id,
            "image": image,
            "image_meta_dict": {
                "affine": affine,
                "spacing": spacing,
            },
            "scales": self.config.scales,
            "base_radius": kwds.get("base_radius", None),
        }
        data = self.infer_transform(data)

        for seed in self.seeds_pool:
            self.logger.warning(f"Use seed {seed}")
            seed_point = torch.tensor(seed).to(self.device)
            self.track_from_seed(data, seed_point, **kwds)
        self.centerline.post_process()
        self.logger.warning("Finished.")

        self.write_centerline_graph(
            os.path.join(kwds["save_dir"], f"{case_id}.gml.gz"), False)
        self.logger.warning("Saving complete.")


class WaveFrontSkeletonSeeds(WaveFrontTrackingPipeline):
    """
    Adapt the wave-front propagation tracking pipeline to fixed-seeds init
    scenario. Only run_tracking() and update_seeds() function is changed.
    """

    def update_seeds(self, segment: str, affine: torch.Tensor, **kwds):
        """ Update seeds following original SIRE paper. """

        current_track = self.centerline.centerline_points.to(self.device)
        # During initialization, all skeleton points are added to the queue.
        if len(current_track) == 0:
            self.logger.warning("Loading segmentation %s..." % segment)
            mask = sitk.GetArrayFromImage(
                sitk.ReadImage(segment, sitk.sitkUInt8)).T
            skel = skeletonize(mask, method="lee")  # Skeletonize
            kernel = ndimage.generate_binary_structure(3, 3)  # Conv kernel
            bif_indicator = ndimage.convolve(
                skel, kernel, mode="constant", cval=0.0) * skel
            skel[bif_indicator > 3] = 0  # Get dis-connected segments

            img_coords = torch.from_numpy(
                np.argwhere(skel)).view(-1, 3).to(self.device)
            phy_coords: torch.Tensor = transform_points(
                img_coords, affine.to(self.device))
            self.seeds_pool = phy_coords.tolist()

            self.counter = 1  # Counter for early termination
        # Remove visited seeds from the pool using radius information
        else:
            new_seeds = []
            if self.counter >= kwds["max_seeds"]:
                self.seeds_pool = []  # Clear after certain number of updates
                self.logger.warning("0 SEEDs Remaining.")
                return
            # NOTE: A new version of seed selection without for loop.
            r = self.centerline.centerline_r.to(self.device)  # (N, )
            remain_seeds = torch.tensor(self.seeds_pool).to(self.device)
            M = remain_seeds.unsqueeze(1)  # (M, 0, 3)
            N = current_track.unsqueeze(0)  # (0, N, 3)
            distMat = torch.linalg.norm(M - N, dim=-1, keepdims=False)
            r = r.clone().view(1, -1).expand(M.shape[0], -1)
            boolMat = torch.all(distMat >= r, dim=-1, keepdim=False)  # (M, )
            new_seeds = remain_seeds[torch.argwhere(boolMat).squeeze()]
            self.seeds_pool = new_seeds.view(-1, 3).tolist()
            self.counter += 1

        self.logger.warning("%d SEEDs Remaining." % len(self.seeds_pool))

    def run_tracking(
        self,
        case_id: str,
        image_name: str,
        **kwds,
    ) -> Dict[List[int | float], Any]:
        """ Run segmentator inference on one image. """

        image, affine, spacing = self.init_on_new_image(image_name, **kwds)
        kwds.update({"affine": affine, })

        # Track from every seed points available, and get centerline tree
        self.update_seeds(**kwds)

        # Get pre-transformed data
        data = {
            "id": case_id,
            "image": image,
            "image_meta_dict": {
                "affine": affine,
                "spacing": spacing,
            },
            "scales": self.config.scales,
            "base_radius": kwds.get("base_radius", None),
        }
        data = self.infer_transform(data)

        while self.seeds_pool:
            seed = self.seeds_pool.pop(0)
            self.logger.warning(f"Use seed {seed}")
            seed_point = torch.tensor(seed).to(self.device)
            self.track_from_seed(data, seed_point, **kwds)
            self.update_seeds(**kwds)
        self.centerline.post_process()
        self.logger.warning("Finished.")

        self.write_centerline_graph(
            os.path.join(kwds["save_dir"], f"{case_id}.gml.gz"), False)
        self.logger.warning("Saving complete.")


class WaveFrontFixToSkeletonSeeds(WaveFrontTrackingPipeline):
    """
    Adapt the wave-front propagation tracking pipeline to fixed-seeds then skel
    eton seeds init scenario. Only run_tracking() and update_seeds() function
    is changed.
    """

    def update_seeds(
        self,
        seeds: List,
        segment: str,
        affine: torch.Tensor,
        **kwds,
    ):
        """ Update seeds from both given seeds and prediction mask. """

        current_track = self.centerline.centerline_points.to(self.device)
        # During initialization, all skeleton points are added to the queue.
        if len(current_track) == 0:
            self.logger.warning("Loading segmentation %s..." % segment)
            mask = sitk.GetArrayFromImage(
                sitk.ReadImage(segment, sitk.sitkUInt8)).T
            skel = skeletonize(mask, method="lee")  # Skeletonize
            kernel = ndimage.generate_binary_structure(3, 3)  # Conv kernel
            bif_indicator = ndimage.convolve(
                skel, kernel, mode="constant", cval=0.0) * skel
            skel[bif_indicator > 3] = 0  # Get dis-connected segments

            img_coords = torch.from_numpy(
                np.argwhere(skel)).view(-1, 3).to(self.device)
            phy_coords: torch.Tensor = transform_points(
                img_coords, affine.to(self.device))
            phy_coords = phy_coords.tolist()
            self.seeds_pool = copy.deepcopy(seeds)
            self.seeds_pool.extend(phy_coords)  # Append phy coords to seeds

            self.counter = 1  # Counter for early termination
        # Remove visited seeds from the pool using radius information
        else:
            new_seeds = []
            if self.counter >= kwds["max_seeds"]:
                self.seeds_pool = []  # Clear after certain number of updates
                self.logger.warning("0 SEEDs Remaining.")
                return
            r = self.centerline.centerline_r.to(self.device)  # (N, )
            remain_seeds = torch.tensor(self.seeds_pool).to(self.device)
            if len(remain_seeds) <= 1:
                new_seeds = remain_seeds
            else:
                M = remain_seeds.unsqueeze(1)  # (M, 0, 3)
                N = current_track.unsqueeze(0)  # (0, N, 3)
                distMat = torch.linalg.norm(M - N, dim=-1, keepdims=False)
                r = r.clone().view(1, -1).expand(M.shape[0], -1)
                boolMat = torch.all(distMat >= r, dim=-1, keepdim=False)
                new_seeds = remain_seeds[torch.argwhere(boolMat).squeeze()]
            self.seeds_pool = new_seeds.view(-1, 3).tolist()
            self.counter += 1

        self.logger.warning("%d SEEDs Remaining." % len(self.seeds_pool))

    def run_tracking(
        self,
        case_id: str,
        image_name: str,
        **kwds,
    ) -> Dict[List[int | float], Any]:
        """ Run segmentator inference on one image. """

        self._build_stopping_criteria(**kwds)  # Re-initialize stopping config

        image, affine, spacing = self.init_on_new_image(image_name, **kwds)
        kwds.update({"affine": affine, })

        # Track from every seed points available, and get centerline tree
        self.update_seeds(**kwds)

        # Get pre-transformed data
        data = {
            "id": case_id,
            "image": image,
            "image_meta_dict": {
                "affine": affine,
                "spacing": spacing,
            },
            "scales": self.config.scales,
            "base_radius": kwds.get("base_radius", None),
        }
        data = self.infer_transform(data)

        while self.seeds_pool:
            seed = self.seeds_pool.pop(0)
            self.logger.warning(f"Use seed {seed}")
            seed_point = torch.tensor(seed).to(self.device)
            self.track_from_seed(data, seed_point, **kwds)
            self.update_seeds(**kwds)
        self.centerline.post_process()
        self.logger.warning("Finished.")

        self.write_centerline_graph(
            os.path.join(kwds["save_dir"], f"{case_id}.gml.gz"), False)
        self.logger.warning("Saving complete.")


class AdaSireWaveFrontFixSeeds(WaveFrontFixSeeds):

    def __init__(self, config, log_dir):

        super().__init__(config, log_dir)
        self.direction_transform = Compose([
            AdaSireSampleAndPredict(config),
            AdaSirePredictionNormalize(config),
            IcoSphereLocalMaxSolver(config),
        ])
        self.infer_transform = Compose([
            InferImgPreTransform(config),
            BuildMultiScaleGraphStructure(config),
            FillSamplingCoords(config),
            BuildForGEMGCN(keys=["graph"]),
        ])


class AdaSireWaveFrontSkeletonSeeds(WaveFrontSkeletonSeeds):

    def __init__(self, config, log_dir):

        super(WaveFrontSkeletonSeeds, self).__init__(config, log_dir)
        self.direction_transform = Compose([
            AdaSireSampleAndPredict(config),
            AdaSirePredictionNormalize(config),
            IcoSphereLocalMaxSolver(config),
        ])
        self.infer_transform = Compose([
            InferImgPreTransform(config),
            BuildMultiScaleGraphStructure(config),
            FillSamplingCoords(config),
            BuildForGEMGCN(keys=["graph"]),
        ])


class AdaSireWaveFrontFixToSkeletonSeeds(WaveFrontFixToSkeletonSeeds):

    def __init__(self, config, log_dir):

        super(WaveFrontFixToSkeletonSeeds, self).__init__(config, log_dir)
        self.direction_transform = Compose([
            AdaSireSampleAndPredict(config),
            AdaSirePredictionNormalize(config),
            IcoSphereLocalMaxSolver(config),
        ])
        self.infer_transform = Compose([
            InferImgPreTransform(config),
            BuildMultiScaleGraphStructure(config),
            FillSamplingCoords(config),
            BuildForGEMGCN(keys=["graph"]),
        ])
