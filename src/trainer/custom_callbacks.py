from pytorch_lightning.callbacks import Callback

from src.utils.load_transforms import LoadImageMaskRTCached


# NOTE: Currently this module does not work well for LoadImageRTCached!
class SharedMemoryCleanUpCallback(Callback):

    def on_exception(self, trainer, pl_module, exception):

        print("Exception occurred: ")
        print(repr(exception))

        print("Cleaning up Shared memory if any...")
        # Guard defensively: the val dataloader may be absent (e.g. when
        # limit_val_batches=0) and some datasets keep LoadImageMask inside a
        # Compose rather than exposing `load_transform`. Never let cleanup raise
        # here, or it masks the original exception.
        for loader in (trainer.train_dataloader, trainer.val_dataloaders):
            dataset = getattr(loader, "dataset", None)
            load_transform = getattr(dataset, "load_transform", None)
            if isinstance(load_transform, LoadImageMaskRTCached):
                load_transform.cleanup()
        print("Finished.")
