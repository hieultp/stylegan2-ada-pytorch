import logging
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
import wandb
from adabelief_pytorch import AdaBelief
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.models import regnet_x_8gf
from torchvision.models.feature_extraction import create_feature_extractor

from lightning_utils import pil_loader, set_debug_apis, split_normalization_params

log = logging.getLogger(__name__)


class MyDataset(VisionDataset):
    def __init__(
        self, root: str, transforms=None, transform=None, target_transform=None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.pair_filenames = [
            [empty, full]
            for empty, full in zip(
                Path(root).joinpath("empty").rglob("*.png"),
                Path(root).joinpath("full").rglob("*.png"),
            )
        ]

    def rand_num(self):
        return torch.rand(1).item()

    def __getitem__(self, index: int):
        first, second = self.pair_filenames[index]
        label = 1.0

        # Augment
        replace_first = self.rand_num() < 0.25
        replace_second = self.rand_num() < 0.25
        swap_place = self.rand_num() < 0.5

        while replace_first and replace_second:
            replace_first = self.rand_num() < 0.25
            replace_second = self.rand_num() < 0.25

        if swap_place:
            first, second = second, first

        if replace_first:
            first = self.pair_filenames[
                torch.randint(len(self.pair_filenames), (1,)).item()
            ][0]
            label = -1.0
        elif replace_second:
            second = self.pair_filenames[
                torch.randint(len(self.pair_filenames), (1,)).item()
            ][1]
            label = -1.0

        first = np.array(pil_loader(first), dtype=np.float32, copy=False) / 255
        second = np.array(pil_loader(second), dtype=np.float32, copy=False) / 255
        image = np.concatenate([first, second], axis=1)
        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.pair_filenames)


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.normal_transform = tvf.ToTensor()

    def setup(self, stage=None) -> None:
        dataset = MyDataset(
            root=self.config.root,
            transform=self.normal_transform,
        )
        self.val_set, self.train_set = random_split(
            dataset, (self.config.val_size, self.config.train_size)
        )
        self.train_set.dataset.transform = tvf.Compose(
            [self.normal_transform, tvf.RandomHorizontalFlip()]
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


class Net(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        self.net = create_feature_extractor(
            resnet, return_nodes={"trunk_output": "feats", "fc": "pred"}
        )

    def forward(self, x):
        extracted = self.net(x)
        feats = extracted["feats"]
        pred = extracted["pred"].tanh()

        return pred, feats


class Measurer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.net = Net(
            regnet_x_8gf(
                pretrained=self.hparams.pretrained,
                progress=True,
                num_classes=1,
            )
        )

    def forward(self, x):
        return self.net(x)

    def _calculate_loss(self, pred, y, feats):
        x1_feats, x2_feats = torch.chunk(feats, 2, dim=-1)
        x1_feats = x1_feats.flatten(1)
        x2_feats = x2_feats.flatten(1)

        soft_margin_loss = F.soft_margin_loss(pred, y.unsqueeze(-1))
        cosine_embedding_loss = F.cosine_embedding_loss(x1_feats, x2_feats, y)
        loss = 1e0 * soft_margin_loss + 1e0 * cosine_embedding_loss
        return {
            "soft_margin_loss": soft_margin_loss,
            "cosine_embedding_loss": cosine_embedding_loss,
            "loss": loss,
        }

    def step(self, batch):
        x, y = batch
        pred, feats = self(x)
        loss_dict = self._calculate_loss(pred, y, feats)
        return y, pred, loss_dict

    def get_log(self, loss_dict, pred, y, state="train"):
        assert state in ["train", "val"]

        logs = {}
        for key, val in loss_dict.items():
            logs[f"{state}/{key}"] = val
        return logs

    def training_step(self, batch, *args, **kwargs):
        y, pred, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, pred, y, state="train")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return loss_dict["loss"]

    def validation_step(self, batch, *args, **kwargs):
        y, pred, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, pred, y, state="val")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return pred, loss_dict

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


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, samples=8) -> None:
        super().__init__()

        self.wandb_logger: WandbLogger = wandb_logger
        self.samples = samples

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log image predictions from the first batch
        if batch_idx == 0:
            n = self.samples
            x, y = batch
            pred, _ = outputs
            images = [img for img in x[:n]]
            captions = [
                f"GT: {y_i.item()} - Pred: {y_pred.item()}"
                for y_i, y_pred in zip(y[:n], pred[:n])
            ]

            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(
                key="val/visualization", images=images, caption=captions
            )

            # Option 2: log images and predictions as a W&B Table
            # columns = ["image", "ground truth", "prediction"]
            # data = [
            #     [wandb.Image(x_i), y_i, y_pred]
            #     for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
            # ]
            # self.wandb_logger.log_table(key="sample_table", columns=columns, data=data)


def train(config):
    config.seed = pl.seed_everything(seed=config.seed, workers=True)

    wandb_logger = WandbLogger(
        project="nsd-bedroom256-similarity",
        log_model=False,
        settings=wandb.Settings(start_method="fork"),
        name=Path.cwd().stem,
    )

    # Create callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(**config.model_ckpt))
    callbacks.append(RichProgressBar(config.refresh_rate))
    callbacks.append(LogPredictionSamplesCallback(wandb_logger))

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

    model = Measurer(config.model)
    datamodule = DataModule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="similarity")
def main(config: DictConfig) -> None:
    log.info("Bedroom 256 metric learner")
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
