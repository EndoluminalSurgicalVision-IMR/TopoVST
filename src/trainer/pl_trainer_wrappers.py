# Self-implementation of COACT tracker model trainer.
import os
import json
from datetime import datetime

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch_geometric.loader import DataLoader

from src.utils.config import BaseConfig
from src.trainer.custom_callbacks import SharedMemoryCleanUpCallback


class PytorchLightningTrainWrapper():

    def __init__(self, config: BaseConfig) -> None:

        self.config = config
        # Define dataset
        self.train_dataset = config.dataset(config=config, stage="train")
        self.val_dataset = config.dataset(config=config, stage="val")

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            persistent_workers=True)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers,
            persistent_workers=True)

        self.pl_module: LightningModule = config.pl_module(**config.pl_config)

        run_id = getattr(config, "run_id", None)
        if config.ckpt is None:
            # Fresh run: timestamped version, suffixed with run_id to avoid
            # same-second collisions between concurrent runs.
            base_version = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            version = f"{base_version}_{run_id}" if run_id else base_version
        else:
            # Resume: reuse the original log directory verbatim (it already
            # encodes the run_id).
            version = config.ckpt.split("/")[-3]
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=f"{config.task_name}_{self.config.dataset_name}",
            version=version,
        )
        modelckpt = ModelCheckpoint(every_n_train_steps=100, save_top_k=-1)
        cleanup = SharedMemoryCleanUpCallback()
        self.pl_trainer = Trainer(logger=logger, callbacks=[modelckpt, cleanup],
                                  **config.trainer)

    def train(self):

        if not os.path.exists(self.pl_trainer.logger.log_dir):
            os.makedirs(self.pl_trainer.logger.log_dir)
        with open(os.path.join(self.pl_trainer.logger.log_dir, "config.json"), "w") as f:
            json.dump(self.config.print_self(), f, indent=4)
        with open(os.path.join(self.pl_trainer.logger.log_dir, "readme.txt"), "w") as f:
            f.write(self.config.aim)

        self.pl_trainer.fit(
            self.pl_module, self.train_dataloader, self.val_dataloader,
            ckpt_path=self.config.ckpt)
