import sys
import time
import logging
from typing import Dict, Tuple, List, Any

import torch
import numpy as np
import networkx as nx
import SimpleITK as sitk
from logdecorator import log_on_start, log_on_end

from src.utils.geometry import get_affine
from src.utils.config import BaseConfig
from src.tester.stopping_criterions import (
    StoppingCriterionBase,
    OutOfBoundaryStoppingCriterion,
)
from src.tester.tracked_centerline import TrackedCenterline


class TrackerInferenceBase(object):
    """
    Base class for tracker inference.
    Args:
        config(BaseConfig): A config object describing tracking configuration.
        log_dir(str): A file location to store tracking logs and results.
    """

    def __init__(self, config: BaseConfig, log_dir: str) -> None:

        self.config = config

        # Initialize logger
        self.logger = logging.getLogger("TrackerInference")
        logging.basicConfig(
            filename=f"{log_dir}/infer_{time.strftime('%Y%m%d_%H%M%S')}.log",
            filemode="w",
            level=logging.INFO,
            format="(%(asctime)s)(%(levelname)s): %(message)s")
        s_handler = logging.StreamHandler(sys.stdout)
        s_handler.setLevel(logging.WARNING)
        s_handler.setFormatter(logging.Formatter(
            "(%(asctime)s)(%(levelname)s): %(message)s"))
        self.logger.addHandler(s_handler)
        # Necessary attributes
        self.device = config.device
        self._build_stopping_criteria()

        # Centerline representation
        self.centerline = TrackedCenterline()
        # Seeds list
        self.seeds_pool = []

    def _build_stopping_criteria(self):
        """ Build stopping criteria instances. Default stopping criteria are
        OutOfBoundaryStoppingCriterion. """

        self.stop_criteria: Dict[str, StoppingCriterionBase] = {}

        # Below stopping criterions MUST be included
        oob_criterion = OutOfBoundaryStoppingCriterion()
        self.stop_criteria.update(
            {
                "OutOfBoundary": oob_criterion,  # Updates on every run.
            }
        )

    def check_stopping_criterions(self, **kwds) -> Dict[str, bool]:
        """ Evaluates list of stopping criterions to check whether tracking
        should terminate. """

        return {
            name: cls_instance(**kwds)
            for name, cls_instance in self.stop_criteria.items()
        }

    def _get_triggered_criterion(
        self,
        criterions_state: Dict[str, bool],
        iteration: int,
    ) -> List[str]:
        """Checks which stopping criterion triggered - for logging sake."""

        for name, state in criterions_state.items():
            if state:
                self.logger.warning(
                    f"STOP: {iteration} iteration. "
                    f"{str(self.stop_criteria[name])}"
                )
            else:
                pass

    def move_one_step(
        self,
        position: torch.Tensor,
        direction: torch.Tensor,
        stepsize: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Move one step along given direction. """

        orient = direction.to(self.device).clone()
        pos = position.to(self.device).clone()

        # Stretch direction tensor to unit length.
        orient /= torch.linalg.norm(orient)
        new_pos = pos + orient * stepsize

        return new_pos.squeeze(), (orient * stepsize).squeeze()

    def update_centerline(
        self,
        position: torch.Tensor | np.ndarray | List,
        parent_node: int,
        **kwds,
    ) -> int:
        """ Update centerline tree with point, its parent and properties to
        store. Return the inserted node index. """

        node_ind = self.centerline.add_point(position, parent_node, **kwds)

        return node_ind

    def update_stopping(self, name: str, **kwds):
        """
        Update specific stopping criterion with keyword arguments.
        """

        criterion = self.stop_criteria.get(name, None)
        if criterion is None:
            self.logger.error(f"{name} is not in current Criterion pool!")
            return
        criterion.update(**kwds)

    @log_on_start(logging.WARNING, "Init on {image_name!r}.")
    def init_on_new_image(self, image_name: str, **kwds):
        """ Initialize tracking process on new image, NIFTI. """

        # Load image from path
        itk_image = sitk.ReadImage(image_name)  # LPS coordinate system
        affine, spacing = get_affine(itk_image)
        image = sitk.GetArrayFromImage(itk_image)

        # Update stopping criterions
        self.update_stopping(
            "OutOfBoundary", shape=image.T.shape, affine=affine)

        # Clear up enture centerline node index and graph.
        self.centerline.reset()

        return torch.from_numpy(image), affine, spacing

    def write_centerline_graph(self, save_path: str, simple_graph: bool = True):
        """ Save currently tracked centerline to GML graph data. """

        if simple_graph:
            nx.write_gml(self.centerline.centerline_graph_simple_nx,
                         save_path, lambda x: str(x))
        else:
            nx.write_gml(self.centerline.centerline_graph_nx,
                         save_path, lambda x: str(x))

    def run_tracking(self, case_id: str, image_name: str, **kwds):

        raise NotImplementedError("Tracking function not implemented.")
