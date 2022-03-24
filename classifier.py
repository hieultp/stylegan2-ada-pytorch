import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as tvf
import wandb
from adabelief_pytorch import AdaBelief
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19_bn

from lightning_utils import set_debug_apis, split_normalization_params

log = logging.getLogger(__name__)


class MyDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        label_smoothing=0.0,
    ):
        super().__init__(root, transform, target_transform)
        self.label_smoothing = label_smoothing

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        if self.label_smoothing > 0.0:
            target = abs(target - self.label_smoothing)
        return sample, target


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.normal_transform = tvf.ToTensor()

    def setup(self, stage=None) -> None:
        dataset = MyDataset(
            root=self.config.root,
            transform=self.normal_transform,
            label_smoothing=self.config.label_smoothing,
        )
        self.val_set, self.train_set = random_split(
            dataset, (self.config.val_size, self.config.train_size)
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

    def _calculate_loss(self, logit, y):
        y = y.unsqueeze(-1)
        loss = self.criterion(logit, y.to(logit.dtype))
        return {"loss": loss}

    def step(self, batch):
        x, y = batch
        logit = self(x)
        loss_dict = self._calculate_loss(logit, y)
        return logit, y, loss_dict

    def get_log(self, loss_dict, logit, y, state="train"):
        assert state in ["train", "val"]

        y = y.unsqueeze(-1)
        if y.is_floating_point():
            y = y.round().to(torch.int64)
        y_hat = logit.sigmoid()
        logs = {"train": self.train_metrics, "val": self.val_metrics}[state](y_hat, y)
        for key, val in loss_dict.items():
            logs[f"{state}/{key}"] = val
        return logs

    def training_step(self, batch, *args, **kwargs):
        logit, y, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, logit, y, state="train")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return loss_dict["loss"]

    def validation_step(self, batch, *args, **kwargs):
        logit, y, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, logit, y, state="val")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return None

    def configure_optimizers(self):
        optim_ops = self.hparams.optimizer
        if optim_ops.norm_weight_decay is None:
            parameters = self.parameters()
        else:
            param_groups = split_normalization_params(self)
            wd_groups = [optim_ops.norm_weight_decay, optim_ops.weight_decay]
            parameters = [
                {"params": p, "weight_decay": w}
                for p, w in zip(param_groups, wd_groups)
                if p
            ]
        optimizer = AdaBelief(
            parameters, lr=optim_ops.lr, weight_decay=optim_ops.weight_decay
        )
        return {
            "optimizer": optimizer,
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(
            set_to_none=True
        )  # This is said that can speed up the training process


def train(config):
    config.seed = pl.seed_everything(seed=config.seed, workers=True)

    wandb_logger = WandbLogger(
        project="nsd-bedroom256-boundary",
        log_model=False,
        settings=wandb.Settings(start_method="fork"),
        name=Path.cwd().stem,
    )

    # Create callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(**config.model_ckpt))
    callbacks.append(RichProgressBar(config.refresh_rate))

    OmegaConf.set_struct(config, False)
    strategy = config.trainer.pop("strategy", None)
    OmegaConf.set_struct(config, True)
    if strategy == "ddp":
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
        strategy = DDPPlugin(
            find_unused_parameters=config.ddp_plugin.find_unused_params,
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook
            if config.ddp_plugin.fp16_hook
            else None,
        )

    model = Classifier(config.model)
    datamodule = DataModule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="classification")
def main(config: DictConfig) -> None:
    log.info("Bedroom 256 boundary finder")
    log.info(f"Current working directory : {Path.cwd()}")

    if config.state == "train":
        set_debug_apis(state=False)
        train(config)
    elif config.state == "debug":
        pass
    elif config.state == "test":
        set_debug_apis(state=False)
        pass


if __name__ == "__main__":
    main()
