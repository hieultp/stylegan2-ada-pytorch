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
from pathlib import Path
from typing import Union

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm, trange

import dnnlib
import legacy
from lightning_utils import pil_loader, set_debug_apis

torch.backends.cudnn.benchmark = True
set_debug_apis(state=False)


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
    source_image: np.ndarray,
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
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

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
    optimizer = torch.optim.AdamW(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
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
    target = source_image.transpose(2, 0, 1)
    target = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
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

    return w_out


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


def run(G, device: torch.device, config: DictConfig, source_fname: Union[str, Path]):
    outdir: Path = Path(config["outdir"])

    source_image = pil_loader(source_fname)
    source_image = np.array(source_image, dtype=np.uint8)

    # Optimize projection.
    w_steps = project(
        G=G,
        source_image=source_image,
        num_steps=config["num_steps"],
        device=device,
    )

    # Render debug output: optional video, image, and W vector.
    outdir.mkdir(parents=True, exist_ok=True)

    # Save final frame and W vector.
    if source_image is not None:
        Image.fromarray(source_image, "RGB").save(f"{outdir}/source.png")
    w = w_steps[-1]
    synth_image = G.synthesis(
        w.unsqueeze(0).repeat([1, G.num_ws, 1]), noise_mode="const"
    )
    synth_image = tensor_to_np_img(synth_image)
    Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.save(f"{outdir}/proj.npy", w.unsqueeze(0).cpu().numpy())

    # Moved the video to the end so we can look at images while these videos take time to compile
    if getattr(config, "save_video", False):
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=24, codec="libx264", bitrate="16M"
        )
        for idx, w in enumerate(w_steps):
            synth_image = G.synthesis(w.unsqueeze(0), noise_mode="const")
            synth_image = tensor_to_np_img(synth_image)
            if idx == 0:
                ref_image = source_image if source_image is not None else synth_image
            video.append_data(np.concatenate([ref_image, synth_image], axis=1))
        video.close()


@hydra.main(config_path="configs", config_name="large_projection")
def main(config: DictConfig) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load networks.
    print(f'Loading networks from "{config["network_pkl"]}"...')
    with dnnlib.util.open_url(config["network_pkl"]) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    print(f'Output folder is "{config["outdir"]}"')
    source_dir = Path(config["source_dir"])
    pairs = random.sample(
        [
            [empty, full]
            for empty, full in zip(
                source_dir.joinpath("empty").rglob("*.png"),
                source_dir.joinpath("full").rglob("*.png"),
            )
        ],
        k=config["num_pairs"],
    )
    for pair in tqdm(pairs):
        empty, full = pair
        current_outdir = config["outdir"]
        config["outdir"] = os.path.join(
            current_outdir, "empty", "-".join(empty.parts[-2:]).replace(".png", "")
        )
        run(G, device, config, empty)
        config["outdir"] = os.path.join(
            current_outdir, "full", "-".join(full.parts[-2:]).replace(".png", "")
        )
        run(G, device, config, full)


if __name__ == "__main__":
    main()
