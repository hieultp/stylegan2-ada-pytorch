import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
from torchvision.models import efficientnet_b2

import dnnlib
from lightning_utils import pil_loader, set_debug_apis, split_normalization_params

log = logging.getLogger(__name__)


class MyDataset(VisionDataset):
    def __init__(
        self, root: str, transforms=None, transform=None, target_transform=None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.filenames = [path for path in Path(root).rglob("*.png")]

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image = np.array(pil_loader(filename), dtype=np.float32, copy=False)
        image = (image / 255) * 2 - 1  # [-1, 1]
        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        return image

    def __len__(self) -> int:
        return len(self.filenames)


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


class Encoder(nn.Module):
    def __init__(self, net, w_avg: torch.Tensor, num_ws=14):
        super().__init__()

        self.net = nn.Sequential(
            net,
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.register_buffer("w_avg", w_avg.view(1, -1))  # [1, w_dim]
        self.num_ws = num_ws

    def forward(self, x):
        w_delta = self.net(x)  # [B, w_dim]
        ws = w_delta + self.w_avg  # [B, w_dim] + [1, w_dim]
        ws = ws.unsqueeze(1).repeat(1, self.num_ws, 1)  # [B, num_ws, w_dim]
        return ws


class Mapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        with open(config.stylegan2_pkl, "rb") as f:
            self.generator = pickle.load(f)["G_ema"].eval().requires_grad_(False)
        self.encoder = torch.jit.script(
            Encoder(
                efficientnet_b2(
                    pretrained=self.hparams.pretrained,
                    progress=True,
                    num_classes=self.generator.w_dim,
                ),
                w_avg=self.generator.mapping.w_avg.detach().clone(),
                num_ws=self.generator.num_ws,
            )
        )

        # Load VGG16 feature detector.
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        ws = self.encoder(x)
        synth_images = self.generator.synthesis(ws, noise_mode="const")
        return synth_images

    def _calculate_loss(self, synth_images, real_images):
        mse_loss = self.mse_loss(synth_images, real_images)

        real_images = (real_images + 1) * (255 / 2)
        synth_images = (synth_images + 1) * (255 / 2)
        target_features = self.vgg16(
            real_images, resize_images=False, return_lpips=True
        )
        synth_features = self.vgg16(
            synth_images, resize_images=False, return_lpips=True
        )
        lpips_loss = (target_features - synth_features).square().sum()

        loss = 1e-0 * mse_loss + 8e-1 * lpips_loss
        return {
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
            "loss": loss,
        }

    def step(self, batch):
        real_images = batch
        synth_images = self(real_images)
        loss_dict = self._calculate_loss(synth_images, real_images)
        return synth_images, real_images, loss_dict

    def get_log(self, loss_dict, synth_images, real_images, state="train"):
        assert state in ["train", "val"]

        logs = {}
        for key, val in loss_dict.items():
            logs[f"{state}/{key}"] = val
        return logs

    def training_step(self, batch, *args, **kwargs):
        synth_images, real_images, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, synth_images, real_images, state="train")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return loss_dict["loss"]

    def validation_step(self, batch, *args, **kwargs):
        synth_images, real_images, loss_dict = self.step(batch)
        logs = self.get_log(loss_dict, synth_images, real_images, state="val")
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            sync_dist=self.hparams.sync_dist,
        )
        return synth_images, loss_dict

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
            real_images = batch
            synth_images, _ = outputs
            images = [
                torch.cat([synth, real], dim=-1)
                for synth, real in zip(synth_images[:n], real_images[:n])
            ]
            captions = ["Proj - GT"] * n

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
        project="nsd-bedroom256-projection",
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

    model = Mapper(config.model)
    datamodule = DataModule(config.dataset)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=strategy,
        **config.trainer,
    )

    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="projection")
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
