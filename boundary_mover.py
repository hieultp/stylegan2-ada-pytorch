# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import hydra
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from pykeops.torch import LazyTensor
from pytorch_lightning import seed_everything
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import normalize
from tqdm import trange

import dnnlib
import legacy
from autoencoder_f2e import MapperF2E
from classifier import Classifier
from lightning_utils import pil_loader, set_debug_apis
from metric_learner import Measurer

torch.backends.cudnn.benchmark = True
set_debug_apis(state=False)


def kmeans_clustering(x, K=10, n_iters=100):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].detach().clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for _ in range(n_iters):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c = c / (Ncl + 1)  # compute the average and add one to advoid divide by zero

    return cl, c


def imagenet_normalize(tensor: torch.Tensor):
    return normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def init_w(G, seed, w_avg_samples, device: torch.device):
    # Compute w stats.
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].to(torch.float32)  # [N, 1, C]
    w_avg = w_samples.mean(dim=0, keepdim=True)  # [1, 1, C]
    w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    return w_avg, w_std


def project(
    G,
    target: torch.Tensor,  # [1,C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    seed=None,
    num_steps=500,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    device: torch.device,
):
    w_avg, w_std = init_w(G, seed, w_avg_samples, device)

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

    w_opt = w_avg.detach().clone().requires_grad_(True)
    w_out = torch.zeros(
        [num_steps + 1] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    w_out[0] = w_opt.detach()[0]
    optimizer = AdaBelief(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
        print_change_log=False,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Load VGG16 feature detector.
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    if target.shape[2] > 256:
        target = F.interpolate(target, size=(256, 256), mode="area")
    target_features = vgg16(target, resize_images=False, return_lpips=True)

    pbar = trange(num_steps, miniters=50, leave=False)
    for step in pbar:
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.num_ws, 1])
        synth_image: torch.Tensor = G.synthesis(ws, noise_mode="const")
        synth_image = (synth_image + 1) * 0.5

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_image.shape[2] > 256:
            synth_image = F.interpolate(synth_image, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(
            synth_image * 255, resize_images=False, return_lpips=True
        )
        lpips = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = lpips + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        pbar.set_postfix_str(
            f"lpips: {lpips:<4.2f}, loss: {float(loss):<4.2f}", refresh=False
        )

        # Save W for each optimization step.
        w_out[step + 1] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out, w_std


def translate(
    G,
    classifier: Classifier,
    measurer: Measurer,
    autoencoder_f2e: MapperF2E,
    deeplabv3: nn.Module,
    ssim: Callable,
    kmeans: Callable,
    target_cls: int,
    source_image: np.ndarray,
    *,
    seed=None,
    num_steps=1000,
    w_avg_samples=4,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    bce_weight=1e0,
    sim_weight=1e0,
    ae_weight=1e0,
    seg_weight=1e0,
    color_weight=1e0,
    device: torch.device,
):
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    if source_image is not None:
        source_image = source_image.transpose(2, 0, 1)
        source_image = torch.tensor(
            source_image, dtype=torch.float32, device=device
        ).unsqueeze(0)
        w_out_project, w_std = project(G, source_image, seed=seed, device=device)
        w_opt = w_out_project[-1].view(1, 1, -1)
        source_image = source_image / 255
    else:
        # Compute w stats.
        print(f"Computing W midpoint and stddev using {w_avg_samples} samples...")
        w_out_project = None
        w_opt, w_std = init_w(G, seed, w_avg_samples, device)

    w_opt = w_opt.detach().clone().requires_grad_(True)
    w_out = torch.zeros(
        [num_steps + 1] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    w_out[0] = w_opt.detach()[0]
    optimizer = AdaBelief(
        [w_opt],
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
        print_change_log=False,
    )

    # Prepare target
    target_sim = torch.tensor([1.0], dtype=torch.float32, device=device)
    target_cls = torch.tensor([target_cls], dtype=torch.float32, device=device)
    target_ssim = torch.tensor(0.6293, dtype=torch.float32, device=device)

    loss_dict = {
        "step": [],
        "loss": [],
        "bce": [],
        "sim": [],
        "ae": [],
        "seg": [],
        "color": [],
    }

    pbar = trange(num_steps, miniters=50, leave=False)
    for step in pbar:
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.num_ws, 1])
        synth_image: torch.Tensor = G.synthesis(ws, noise_mode="const")
        synth_image = (synth_image + 1) * 0.5

        if step == 0:
            reference_image = synth_image.detach().clone()
            if kmeans is not None:
                _, reference_palette = kmeans(
                    rearrange(reference_image, "1 C H W -> (1 H W) C").contiguous()
                )  # FIXME: Support batching

        bce_loss = 0.0
        if classifier is not None:
            # BCELoss to guide the latent based on the classifier
            logit: torch.Tensor = classifier(synth_image)
            bce_loss = classifier._calculate_loss(logit, target_cls)["loss"]

        sim_loss = 0.0
        if measurer is not None:
            concat_images = torch.cat([reference_image, synth_image], dim=-1)
            sim_score, feats = measurer(concat_images)
            sim_loss += measurer._calculate_loss(sim_score, target_sim, feats)["loss"]
        if ssim is not None:
            sim_loss += F.mse_loss(ssim(synth_image, reference_image), target_ssim)

        ae_loss = 0.0
        if autoencoder_f2e is not None:
            _, reference_image_hat = autoencoder_f2e(synth_image)
            ae_loss += autoencoder_f2e._calculate_loss(
                reference_image_hat, reference_image
            )["loss"]

        seg_loss = 0.0
        if deeplabv3 is not None:
            segments: torch.Tensor = deeplabv3(imagenet_normalize(synth_image))["out"]
            background_mask = segments.argmax(dim=1, keepdim=True) == 0
            seg_loss += (
                F.mse_loss(synth_image, reference_image, reduction="none")
                * background_mask
            ).mean()

        color_loss = 0.0
        if kmeans is not None:
            _, current_palette = kmeans(
                rearrange(synth_image, "1 C H W -> (1 H W) C").contiguous()
            )
            color_loss += F.mse_loss(current_palette, reference_palette)

        loss = (
            bce_weight * bce_loss
            + sim_weight * sim_loss
            + ae_weight * ae_loss
            + seg_weight * seg_loss
            + color_weight * color_loss
        )

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        loss_dict["step"].append(step)
        loss_dict["loss"].append(loss.item())
        loss_dict["bce"].append(bce_loss.item())
        loss_dict["sim"].append(sim_loss.item())
        loss_dict["ae"].append(ae_loss.item())
        loss_dict["seg"].append(seg_loss.item())
        loss_dict["color"].append(color_loss.item())

        pbar_postfix_strs = [f"loss={loss.item():<4.2f}"]
        if classifier is not None:
            curr_cls = logit.sigmoid()
            pbar_postfix_strs.extend(
                [
                    f"curr_cls={curr_cls.item():<4.2f}",
                    f"desired_cls={target_cls.item():.0f}",
                ]
            )
        pbar.set_postfix_str(
            ", ".join(pbar_postfix_strs),
            refresh=False,
        )

        # Save W for each optimization step.
        w_out[step + 1] = w_opt.detach()[0]

    if w_out_project is not None:
        w_out = torch.cat([w_out_project, w_out])
    return w_out.repeat([1, G.num_ws, 1]), loss_dict


def setup_networks(config: DictConfig, device: torch.device):
    # Load networks.
    print(f'Loading networks from "{config["network_pkl"]}"...')
    with dnnlib.util.open_url(config["network_pkl"]) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    classifier = None
    if getattr(config, "classifier_pkl", ""):
        # Load classifier
        print(f'Loading classifier from "{config["classifier_pkl"]}"...')
        with dnnlib.util.open_url(config["classifier_pkl"]) as fp:
            classifier = Classifier.load_from_checkpoint(fp).eval().to(device)
            classifier.freeze()

    measurer = None
    if getattr(config, "measurer_pkl", ""):
        # Load measurer
        print(f'Loading measurer from "{config["measurer_pkl"]}"...')
        with dnnlib.util.open_url(config["measurer_pkl"]) as fp:
            measurer = Measurer.load_from_checkpoint(fp).eval().to(device)
            measurer.freeze()

    autoencoder_f2e = None
    if getattr(config, "autoencoder_f2e_pkl", ""):
        # Load measurer
        print(f'Loading autoencoder_f2e from "{config["autoencoder_f2e_pkl"]}"...')
        with dnnlib.util.open_url(config["autoencoder_f2e_pkl"]) as fp:
            autoencoder_f2e = MapperF2E.load_from_checkpoint(fp).eval().to(device)
            autoencoder_f2e.freeze()

    deeplabv3 = None
    if getattr(config, "use_deeplabv3", False):
        print("Loading DeepLabV3 from torchvision")
        deeplabv3 = deeplabv3_resnet50(pretrained=True).eval().to(device)
        for param in deeplabv3.parameters():
            param.requires_grad_(False)

    ssim = None
    if getattr(config, "use_ssim", False):
        print("Use structure similarity index in the translation")
        ssim = partial(structural_similarity_index_measure, data_range=1.0)

    kmeans = None
    if getattr(config, "use_kmeans", False):
        print("Use K-Means for color preservation in the translation")
        kmeans = partial(kmeans_clustering, K=8)

    assert classifier is not None or autoencoder_f2e is not None
    return {
        "G": G,
        "classifier": classifier,
        "measurer": measurer,
        "autoencoder_f2e": autoencoder_f2e,
        "deeplabv3": deeplabv3,
        "ssim": ssim,
        "kmeans": kmeans,
    }


def tensor_to_np_img(synth_image: torch.Tensor):
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = (
        synth_image.clip(0, 255)
        .round()
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .to(device="cpu", dtype=torch.uint8)
        .numpy()
    )
    return synth_image


def run(networks: Dict, device: torch.device, config: DictConfig):
    seed = seed_everything(config["seed"])
    fname = f"{seed}-cls-{config['target_cls']}"
    outdir: Path = Path(config["outdir"]) / fname

    if getattr(config, "source_fname", ""):
        source_image = pil_loader(config["source_fname"])
        source_image = np.array(source_image, dtype=np.uint8)
        decorated_image = None
    elif getattr(config, "random_source_fname", False):
        source_dir = Path(config["source_dir"])
        empty, full = random.choice(
            [
                [empty, full]
                for empty, full in zip(
                    source_dir.joinpath("empty").rglob("*.png"),
                    source_dir.joinpath("full").rglob("*.png"),
                )
            ]
        )
        source_image = np.array(pil_loader(empty), dtype=np.uint8)
        decorated_image = pil_loader(full)
    else:
        source_image = None
        decorated_image = None

    # Optimize projection.
    w_steps, loss_dict = translate(
        **networks,
        target_cls=config["target_cls"],
        source_image=source_image,
        seed=seed,
        num_steps=config["num_steps"],
        device=device,
    )

    # Render debug output: optional video, image, and W vector.
    outdir.mkdir(parents=True, exist_ok=True)

    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(f"{outdir}/loss.csv", index=False)
    loss_df.plot(
        x="step", y="loss bce sim ae seg color".split(), legend=True
    ).get_figure().savefig(f"{outdir}/loss.png")

    # Save final frame and W vector.
    if source_image is not None:
        Image.fromarray(source_image, "RGB").save(f"{outdir}/source.png")
        if decorated_image is not None:
            decorated_image.save(f"{outdir}/full.png")
    w = w_steps[-1]
    synth_image = networks["G"].synthesis(w.unsqueeze(0), noise_mode="const")
    synth_image = tensor_to_np_img(synth_image)
    Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.save(f"{outdir}/proj.npy", w.unsqueeze(0).cpu().numpy())

    # Moved the video to the end so we can look at images while these videos take time to compile
    if getattr(config, "save_video", True):
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=24, codec="libx264", bitrate="16M"
        )
        for idx, w in enumerate(w_steps):
            synth_image = networks["G"].synthesis(w.unsqueeze(0), noise_mode="const")
            synth_image = tensor_to_np_img(synth_image)
            if idx == 0:
                ref_image = source_image if source_image is not None else synth_image
            video.append_data(np.concatenate([ref_image, synth_image], axis=1))
        video.close()


@hydra.main(config_path="configs", config_name="boundary_mover")
def main(config: DictConfig) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mode = getattr(config, "mode", "random_seed")
    print(f'Output folder is "{config["outdir"]}"')
    if mode == "random_seed":
        for cfg in config:
            if cfg in [
                "measurer_pkl",
                "autoencoder_f2e_pkl",
            ]:
                config[cfg] = ""
            if "use_" in cfg:
                config[cfg] = False
        networks = setup_networks(config, device)
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        for seed in [random.randint(min_seed_value, max_seed_value) for _ in range(5)]:
            config["seed"] = seed
            run(networks, device, config)
    else:
        for path in config["disable_networks"]:
            config[path] = ""
        networks = setup_networks(config, device)
        if getattr(config, "random_source_fname", False):
            max_seed_value = np.iinfo(np.uint32).max
            min_seed_value = np.iinfo(np.uint32).min
            seeds = [random.randint(min_seed_value, max_seed_value) for _ in range(5)]
        else:
            seeds = [
                int(path[: path.find("-cls-")])
                for path in os.listdir(config["reference_dir"])
            ]
        for seed in seeds:
            config["seed"] = seed
            path = list(Path(config["reference_dir"]).rglob(f"{seed}*"))
            if path and len(path) == 1:
                config["source_fname"] = str(path[0] / "source.png")
            run(networks, device, config)


if __name__ == "__main__":
    main()
