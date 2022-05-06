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
from typing import Dict

import hydra
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import normalize
from tqdm import trange

import dnnlib
import legacy
from autoencoder import Mapper
from autoencoder_f2e import MapperF2E
from classifier import Classifier
from lightning_utils import pil_loader, set_debug_apis
from metric_learner import Measurer

torch.backends.cudnn.benchmark = True
set_debug_apis(state=False)


def imagenet_normalize(tensor: torch.Tensor):
    return normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def project(
    G,
    classifier: Classifier,
    measurer: Measurer,
    autoencoder: Mapper,
    encoded_vec: torch.Tensor,
    autoencoder_f2e: MapperF2E,
    deeplabv3: nn.Module,
    use_ssim: bool,
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
    regularize_noise_weight=1e5,
    sim_weight=1e0,
    ae_weight=1e0,
    segmentation_weight=1e0,
    device: torch.device,
):
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    print(f"Computing W midpoint and stddev using {w_avg_samples} samples...")
    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].to(torch.float32)  # [N, 1, C]
    w_avg = w_samples.mean(dim=0, keepdim=True)  # [1, 1, C]
    w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

    w_opt = w_avg.detach().clone().requires_grad_(True)
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
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

    # Prepare target
    target_sim = torch.tensor([1.0], dtype=torch.float32, device=device)
    target_cls = torch.tensor([target_cls], dtype=torch.float32, device=device)
    target_ssim = torch.tensor(0.6293, dtype=torch.float32, device=device)
    if source_image is not None:
        source_image = source_image.transpose(2, 0, 1)
        source_image = torch.tensor([source_image], dtype=torch.float32, device=device)
        source_image = source_image / 255

    pbar = trange(num_steps, miniters=10)
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
        synth_image = ((synth_image + 1) * 0.5).clip(0, 1)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # if synth_image.shape[2] > 256:
        #     synth_image = F.interpolate(synth_image, size=(256, 256), mode="area")

        if step == 0:
            reference_image = (
                source_image
                if source_image is not None
                else synth_image.detach().clone()
            )
            if autoencoder is not None:
                init_latent, _ = autoencoder(reference_image)
            if deeplabv3 is not None:
                original_segment = deeplabv3(imagenet_normalize(reference_image))[
                    "out"
                ][0:1]

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
        if use_ssim:
            sim_loss += F.mse_loss(
                structural_similarity_index_measure(synth_image, reference_image),
                target_ssim,
            )

        ae_loss = 0.0
        if autoencoder is not None:
            current_latent, _ = autoencoder(synth_image)
            current_vec = init_latent - current_latent
            ae_loss += F.mse_loss(current_vec, encoded_vec)

        if autoencoder_f2e is not None:
            _, reference_image_hat = autoencoder_f2e(synth_image)
            ae_loss += autoencoder_f2e._calculate_loss(
                reference_image_hat, reference_image
            )["loss"]

        segmentation_loss = 0.0
        if deeplabv3 is not None:
            current_segment = deeplabv3(imagenet_normalize(synth_image))["out"][0:1]
            segmentation_loss += F.mse_loss(current_segment, original_segment)

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

        loss = (
            bce_weight * bce_loss
            + regularize_noise_weight * reg_loss
            + sim_weight * sim_loss
            + ae_weight * ae_loss
            + segmentation_weight * segmentation_loss
        )

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        # curr_cls = logit.sigmoid()
        pbar.set_postfix_str(
            # f"curr_cls={curr_cls.item():<4.2f}, "
            # f"desired_cls={target_cls.item():.0f}, "
            f"loss={loss.item():<4.2f}",
            refresh=False,
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.num_ws, 1])


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

    autoencoder = None
    encoded_vec = None
    if getattr(config, "autoencoder_pkl", ""):
        # Load measurer
        print(f'Loading autoencoder from "{config["autoencoder_pkl"]}"...')
        with dnnlib.util.open_url(config["autoencoder_pkl"]) as fp:
            autoencoder = Mapper.load_from_checkpoint(fp).eval().to(device)
            autoencoder.freeze()
        encoded_vec = torch.load(config["encoded_vec"], map_location=device)

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

    use_ssim = getattr(config, "use_ssim", False)
    if use_ssim:
        print("Use structure similarity index in projection")

    assert (classifier is not None or autoencoder_f2e is not None)
    return {
        "G": G,
        "classifier": classifier,
        "measurer": measurer,
        "autoencoder": autoencoder,
        "encoded_vec": encoded_vec,
        "autoencoder_f2e": autoencoder_f2e,
        "deeplabv3": deeplabv3,
        "use_ssim": use_ssim,
    }


def run_projection(networks: Dict, device: torch.device, config: DictConfig):
    # TODO: project empty/full -> latent space, measure distance between them
    seed = seed_everything(config["seed"])
    fname = f"{seed}-cls-{config['target_cls']}"
    if getattr(config, "source_fname", ""):
        fname = f"{fname}-{Path(config['source_fname']).stem}"
    outdir: Path = Path(config["outdir"]) / fname

    if getattr(config, "source_fname", ""):
        source_image = pil_loader(config["source_fname"])
        source_image = np.array(source_image, dtype=np.uint8)
    else:
        source_image = None

    # Optimize projection.
    projected_w_steps = project(
        **networks,
        target_cls=config["target_cls"],
        source_image=source_image,
        seed=seed,
        num_steps=config["num_steps"],
        device=device,
    )

    # Render debug output: optional video and projected image and W vector.
    outdir.mkdir(parents=True, exist_ok=True)

    def tensor_to_np_img(synth_image):
        synth_image = (synth_image + 1) * (255 / 2)
        return (
            synth_image.permute(0, 2, 3, 1)
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .cpu()
            .numpy()
        )

    # Save final projected frame and W vector.
    if source_image is not None:
        PIL.Image.fromarray(source_image, "RGB").save(f"{outdir}/source.png")
    projected_w = projected_w_steps[-1]
    synth_image = networks["G"].synthesis(projected_w.unsqueeze(0), noise_mode="const")
    synth_image = tensor_to_np_img(synth_image)
    PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.save(f"{outdir}/proj.npy", projected_w.unsqueeze(0).cpu().numpy())

    # Moved the video to the end so we can look at images while these videos take time to compile
    if getattr(config, "save_video", True):
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=24, codec="libx264", bitrate="16M"
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for idx, projected_w in enumerate(projected_w_steps):
            synth_image = networks["G"].synthesis(
                projected_w.unsqueeze(0), noise_mode="const"
            )
            synth_image = tensor_to_np_img(synth_image)
            if idx == 0:
                ref_image = source_image if source_image is not None else synth_image
            video.append_data(np.concatenate([ref_image, synth_image], axis=1))
        video.close()


@hydra.main(config_path="configs", config_name="boundary_mover")
def main(config: DictConfig) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mode = getattr(config, "mode", "random_seed")
    if mode == "random_seed":
        for path in [
            "measurer_pkl",
            "autoencoder_pkl",
            "encoded_vec",
            "autoencoder_f2e_pkl",
        ]:
            config[path] = ""
        config["use_ssim"] = False
        config["use_deeplabv3"] = False
        networks = setup_networks(config, device)
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        for seed in [random.randint(min_seed_value, max_seed_value) for _ in range(20)]:
            config["seed"] = seed
            run_projection(networks, device, config)
    else:
        for path in config["disable_networks"]:
            config[path] = ""
        networks = setup_networks(config, device)
        seeds = [
            int(path[: path.find("-cls-")])
            for path in os.listdir(config["reference_dir"])
        ]
        for seed in seeds:
            config["seed"] = seed
            run_projection(networks, device, config)


if __name__ == "__main__":
    main()
