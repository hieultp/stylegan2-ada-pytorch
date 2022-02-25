import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as tvf
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19_bn

log = logging.getLogger(__name__)


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


class Bedroom256(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.normal_transform = tvf.ToTensor()

    def setup(self, stage=None) -> None:
        dataset = ImageFolder(root=self.config.root, transform=self.normal_transform)
        self.val_set, self.train_set = random_split(
            dataset, (self.config.val_size, 56076 - self.config.val_size),
        )
        self.train_set.dataset.transform = tvf.Compose(
            [tvf.RandomHorizontalFlip(), self.normal_transform]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )


class Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.net = vgg19_bn(
            pretrained=self.hparams.pretrained, progress=True, num_classes=1
        )  # We're doing binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        metrics = MetricCollection([Accuracy(), Precision(), Recall(), F1Score()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        x, y = batch
        logit = self(x)
        y = y.unsqueeze(-1)
        loss = self.criterion(logit, y.to(logit.dtype))
        return logit, y, loss

    def get_log(self, loss, logit, y, state="train"):
        assert state in ["train", "val"]

        y_hat = logit.sigmoid()
        logs = {"train": self.train_metrics, "val": self.val_metrics}[state](y_hat, y)
        logs[f"{state}/loss"] = loss
        return logs

    def training_step(self, batch, *args, **kwargs):
        logit, y, loss = self.step(batch)
        logs = self.get_log(loss, logit, y, state="train")
        self.log_dict(
            logs, on_step=True, on_epoch=True, sync_dist=self.hparams.sync_dist,
        )
        return loss

    def validation_step(self, batch, *args, **kwargs):
        logit, y, loss = self.step(batch)
        logs = self.get_log(loss, logit, y, state="val")
        self.log_dict(
            logs, on_step=True, on_epoch=True, sync_dist=self.hparams.sync_dist,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.hparams.lr_scheduler
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.hparams.monitor,
            },
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(
            set_to_none=True
        )  # This is said that can speed up the training process


def train(config):
    set_debug_apis(state=False)
    pl.seed_everything(seed=config.seed, workers=True)

    wandb_logger = WandbLogger(
        project="nsd-bedroom256-boundary",
        log_model=False,
        settings=wandb.Settings(start_method="fork"),
    )

    if config.trainer.strategy == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have

        # TODO: Currently only handles gpus = -1 or an int number
        if config.trainer.gpus == -1:
            config.trainer.gpus = torch.cuda.device_count()

        num_nodes = getattr(config.trainer, "num_nodes", 1)
        total_gpus = max(1, config.trainer.gpus * num_nodes)
        config.dataset.batch_size = int(config.dataset.batch_size / total_gpus)
        config.dataset.num_workers = int(config.dataset.num_workers / total_gpus)
    #     ddp_plugin = DDPPlugin(
    #         find_unused_parameters=config.ddp_plugin.find_unused_params,
    #         gradient_as_bucket_view=True,
    #         # ddp_comm_hook=default.fp16_compress_hook if self.config.ddp_plugin.fp16_hook else None,
    #     )
    # else:
    #     ddp_plugin = None

    # Create callbacks
    ckpt_callback = ModelCheckpoint(**config.model_ckpt)
    pbar_callback = RichProgressBar(config.refresh_rate)

    model = Classifier(config.model)
    datamodule = Bedroom256(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_callback, pbar_callback],
        # plugins=[ddp_plugin],
        **config.trainer,
    )

    wandb_logger.watch(model)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    log.info("Bedroom 256 boundary finder")
    log.info(f"Current working directory : {Path.cwd()}")

    if config.state == "train":
        train(config)
    elif config.state == "debug":
        pass
    elif config.state == "test":
        set_debug_apis(state=False)
        pass


if __name__ == "__main__":
    main()
