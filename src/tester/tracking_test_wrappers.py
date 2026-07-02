import os
import json
from multiprocessing import Pool
from typing import Dict, Any, List

from src.utils.config import BaseConfig
from src.utils.test_dataloader import TestDataLoader
from src.tester.tracker_inference_base import TrackerInferenceBase


class SeedBasedTrackingWrapper():
    """ Tracking wrapper for seed-based tracking inference. """

    def __init__(
        self,
        config: BaseConfig,
        pipeline: TrackerInferenceBase,
        save_loc: str,
    ):
        """
        Args:
            config (BaseConfig): A config object.
            pipeline (TrackerInferenceBase): A pipeline for tracking.
            save_loc (str): Location to save tracking results.
        """

        self.config = config
        self.prepare_data()
        self.seeds_folder = self.config.seeds_folder
        self.pipeline = pipeline
        self.save_loc = save_loc

    def prepare_data(self):

        eval_loader = TestDataLoader()
        self.images, self.case_ids, self.preds = eval_loader(self.config)
        # Use a list (not a one-shot zip) so callers can iterate the case
        # set more than once — e.g. tracking then a post-processing pass.
        self.cases = list(zip(self.images, self.case_ids, self.preds))

    @property
    def generator(self):
        return iter(self.cases)

    def load_seeds_file(self, case_id: str) -> Dict[str, Any]:
        """ Load seeds and affine matrix in a Dict. """

        # assert self.seeds_folder is not None, "Seeds folder is None!"
        if self.seeds_folder is None:
            return {"seeds": [], }

        json_name = None
        seed_files: List[str] = os.listdir(self.seeds_folder)
        for seed_file in seed_files:
            if seed_file.split("_seeds_")[0] == case_id:
                json_name = os.path.join(self.seeds_folder, seed_file)
                break
        if json_name is None:
            raise FileNotFoundError(
                "Seed file for case %s not found in %s!"
                % (case_id, self.seeds_folder))
        with open(json_name, "r") as f:
            results: Dict[str, Any] = json.load(f)

        return results

    def get_tracking_seeds(self, case_id: str) -> Dict[str, Any]:
        """Return fixed seeds when required by the selected pipeline."""

        if not getattr(self.pipeline, "requires_seed_file", True):
            return {"seeds": []}

        return self.load_seeds_file(case_id)

    def test_one_case(self, args):

        image_path, case_id, pred = args[0], args[1], args[2]
        seeds_d = self.get_tracking_seeds(case_id)
        kwargs: Dict[str, Any] = self.config.infer_config
        kwargs.update({
            "segment": pred,
            "seeds": seeds_d["seeds"],  # For other seeds
            "save_dir": self.save_loc,  # For data saving
        })
        self.pipeline.run_tracking(case_id, image_path, **kwargs)

    def test_parallel_wrapper(self, processes: int = 2):

        input_args = list(self.generator)
        with Pool(processes=processes) as p:
            p.map(self.test_one_case, input_args)

    def test(self) -> None:

        for image_path, case_id, pred in self.generator:
            seeds_d = self.get_tracking_seeds(case_id)
            kwargs: Dict[str, Any] = self.config.infer_config
            kwargs.update({
                "segment": pred,
                "seeds": seeds_d["seeds"],  # For other seeds
                "save_dir": self.save_loc,  # For data saving
            })
            self.pipeline.run_tracking(case_id, image_path, **kwargs)
