from typing import List, Any
import json
import os


class TestDataLoader(object):
    """ A wrapper class for images and other information during inference. """

    def __init__(self):

        pass

    def _prepare_asoca_test_data(self, config):
        """ Given a config object, load image names, mask names and centerline
        names. ASOCA dataset. """

        with open(config.splits, 'r') as f:
            cases: List[str] = json.load(f)[f"{config.phase}"]

        cases.sort()
        case_ids = cases
        images = [os.path.join(config.src_dir, case_id.split("_")[0], "CTCA", "%s.nrrd" % case_id)
                  for case_id in case_ids]
        predictions = [f"{config.preds}/{case_id}.nrrd"
                       for case_id in case_ids]

        return images, case_ids, predictions

    def _prepare_aorta24_test_data(self, config):
        """
        Given a config object, load image names, mask names and centerline
        names. Aorta24 dataset.
        """

        with open(config.splits, "r") as f:
            cases: List[str] = json.load(f)[f"{config.phase}"]

        cases.sort()
        case_ids = cases
        folder = "imagesTr" if config.phase != "test" else "imagesTs"
        images = [os.path.join(config.src_dir, folder, "%s_0000.nii.gz" % case_id)
                  for case_id in case_ids]
        predictions = [f"{config.preds}/{case_id}.nii.gz"
                       for case_id in case_ids]

        return images, case_ids, predictions

    def __call__(self, config: Any):

        match config.dataset_name:

            case "ASOCA":
                return self._prepare_asoca_test_data(config)

            case "Aorta24":
                return self._prepare_aorta24_test_data(config)

            case _:
                raise NotImplementedError(
                    "Preparation for %s not implemented." % config.dataset_name)
