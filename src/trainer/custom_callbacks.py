from pytorch_lightning.callbacks import Callback

from src.utils.load_transforms import LoadImageMaskRTCached


class SharedMemoryCleanUpCallback(Callback):

    def on_exception(self, trainer, pl_module, exception):

        print("Exception occurred: ")
        print(repr(exception))

        print("Cleaning up Shared memory if any...")
        if isinstance(trainer.train_dataloader.dataset.load_transform,
                      LoadImageMaskRTCached):
            trainer.train_dataloader.dataset.load_transform.cleanup()
        if isinstance(trainer.val_dataloaders.dataset.load_transform,
                      LoadImageMaskRTCached):
            trainer.val_dataloaders.dataset.load_transform.cleanup()
        print("Finished.")
