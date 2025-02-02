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

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import trange

import dnnlib
import legacy
from classifier import Classifier
from lightning_utils import set_debug_apis
from metric_learner import Measurer

torch.backends.cudnn.benchmark = True
set_debug_apis(state=False)


def project(
    G,
    classifier: Classifier,
    measurer: Measurer,
    target_image: np.ndarray,
    target_cls: int,
    *,
    seed=None,
    num_steps=1000,
    w_avg_samples=2,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    bce_weight=1e0,
    sim_weight=1e0,
    lpips_weight=1e0,
    regularize_noise_weight=1e5,
    device: torch.device,
):
    assert target_image.shape == (G.img_channels, G.img_resolution, G.img_resolution)

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

    # Prepare target
    target_sim = torch.tensor([1.0], dtype=torch.float32, device=device)
    target_cls = torch.tensor([target_cls], dtype=torch.float32, device=device)
    target_image = torch.tensor([target_image], dtype=torch.float32, device=device)
    if target_image.shape[2] > 256:
        target_image = F.interpolate(target_image, size=(256, 256), mode="area")
    target_features = vgg16(target_image, resize_images=False, return_lpips=True)
    target_image = target_image / 255

    pbar = trange(num_steps)
    for step in pbar:
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.num_ws, 1])
        synth_image: torch.Tensor = G.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_image = ((synth_image + 1) * 0.5).clip(0, 1)
        if synth_image.shape[2] > 256:
            synth_image = F.interpolate(synth_image, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(
            synth_image * 255, resize_images=False, return_lpips=True
        )
        lpips = (target_features - synth_features).square().sum()

        # BCELoss to guide the latent based on the classifier
        logit: torch.Tensor = classifier(synth_image)
        bce_loss = classifier._calculate_loss(logit, target_cls)["loss"]

        if measurer is not None:
            # Similarity loss to guide the latent based on the similarity between the two generated images
            concat_images = torch.cat([target_image, synth_image], dim=-1)
            sim_score, feats = measurer(concat_images)
            sim_loss = measurer._calculate_loss(sim_score, target_sim, feats)["loss"]
        else:
            sim_loss = 0.0

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
            lpips_weight * lpips
            + bce_weight * bce_loss
            + sim_weight * sim_loss
            + reg_loss * regularize_noise_weight
        )

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        curr_cls = logit.sigmoid()
        pbar.set_postfix_str(
            f"curr_cls={curr_cls.item():<4.2f}, "
            f"desired_cls={target_cls.item():.0f}, "
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


# ----------------------------------------------------------------------------


@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--target",
    "target_fname",
    help="Target image file to project to",
    required=True,
    metavar="FILE",
)
@click.option(
    "--classifier",
    "classifier_pkl",
    help="Classifier that has trained on empty/full scenes",
    required=True,
)
@click.option(
    "--measurer",
    "measurer_pkl",
    help="Similarity measurer that has trained on empty/full scenes",
    default="",
    required=True,
)
@click.option(
    "--target-cls",
    help="Target class for the latent code to move",
    type=click.Choice(["0", "1"]),
    default="0",
    show_default=True,
)
@click.option(
    "--num-steps",
    help="Number of optimization steps",
    type=int,
    default=1000,
    show_default=True,
)
@click.option("--seed", help="Random seed", type=int, default=None, show_default=True)
@click.option(
    "--save-video",
    help="Save an mp4 video of optimization progress",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--outdir", help="Where to save the output images", required=True, metavar="DIR"
)
def run_projection(
    network_pkl: str,
    target_fname: str,
    classifier_pkl: str,
    measurer_pkl: str,
    target_cls: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    # TODO: project empty/full -> latent space, measure distance between them
    device = torch.device("cuda")
    seed = seed_everything(seed)
    target_cls = int(target_cls)
    outdir = os.path.join(outdir, f"{seed}-cls-{target_cls}")

    # Load networks.
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)  # type: ignore

    # Load classifier
    print(f'Loading classifier from "{classifier_pkl}"...')
    with dnnlib.util.open_url(classifier_pkl) as fp:
        classifier = Classifier.load_from_checkpoint(fp).eval().to(device)
        classifier.freeze()

    if measurer_pkl:
        # Load measurer
        print(f'Loading measurer from "{measurer_pkl}"...')
        with dnnlib.util.open_url(measurer_pkl) as fp:
            measurer = Measurer.load_from_checkpoint(fp).eval().to(device)
            measurer.freeze()
    else:
        measurer = None

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
    )
    target_img = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    projected_w_steps = project(
        G,
        classifier=classifier,
        measurer=measurer,
        target_img=target_img.transpose(2, 0, 1),
        target_cls=target_cls,
        seed=seed,
        num_steps=num_steps,
        device=device,
    )

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)

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
    target_pil.save(f"{outdir}/target.png")
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
    synth_image = tensor_to_np_img(synth_image)
    PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.savez(f"{outdir}/projected_w.npz", w=projected_w.unsqueeze(0).cpu().numpy())

    # Moved the video to the end so we can look at images while these videos take time to compile
    if save_video:
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=24, codec="libx264", bitrate="16M"
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
            synth_image = tensor_to_np_img(synth_image)
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
