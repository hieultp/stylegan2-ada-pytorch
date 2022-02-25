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
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import trange

import dnnlib
import legacy
from classifier import Classifier


def project(
    G,
    classifier,
    target_cls: torch.Tensor,
    *,
    seed                        = None,
    num_steps                   = 1000,
    w_avg_samples               = 2,
    initial_learning_rate       = 0.1,
    initial_noise_factor        = 0.05,
    lr_rampdown_length          = 0.25,
    lr_rampup_length            = 0.05,
    noise_ramp_length           = 0.75,
    bce_weight                  = 1e0,  # Currently set to 0.0
    regularize_noise_weight     = 1e5,
    verbose                     = False,
    device: torch.device
):

    def logprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)     # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)                    # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    pbar = trange(num_steps)
    for step in pbar:
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = ((synth_images + 1) * 0.5).clip(0, 1)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # BCELoss to guide the latent based on the classifier
        logit: torch.Tensor = classifier(synth_images)
        bce_loss = F.binary_cross_entropy_with_logits(logit, target_cls)

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = bce_weight * bce_loss + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        curr_cls = logit.detach().sigmoid().item()
        pbar.set_postfix_str(
            f'curr_cls={curr_cls:<4.2f}, desired_cls={target_cls.item():.0f}, loss {float(loss):<4.2f}',
            refresh=False,
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',       help='Network pickle filename', required=True)
@click.option('--classifier', 'classifier_pkl', help='Classifier that has trained on empty/full scenes', required=True)
@click.option('--target-cls',                   help='Target class for the latent code to move', type=click.Choice(['0', '1']), default='0', show_default=True)
@click.option('--num-steps',                    help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                         help='Random seed', type=int, default=None, show_default=True)
@click.option('--save-video',                   help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                       help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    classifier_pkl: str,
    target_cls: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    # TODO: project empty/full -> latent space, measure distance between them
    device = torch.device('cuda')
    outdir = os.path.join(outdir, f"{seed}-cls-{target_cls}")
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load networks.
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load classifier
    print(f'Loading classifier from "{classifier_pkl}"...')
    with dnnlib.util.open_url(classifier_pkl) as fp:
        classifier = Classifier.load_from_checkpoint(fp).eval().to(device)
        classifier.freeze()

    # Prepare target class
    target_cls = torch.tensor(
        int(target_cls), dtype=torch.float32, device=device
    ).view(1, 1)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        classifier,
        target_cls=target_cls,
        seed=seed,
        num_steps=num_steps,
        verbose=True,
        device=device,
    )
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)

    def tensor_to_np_img(synth_image):
        synth_image = (synth_image + 1) * (255/2)
        return synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    # Save final projected frame and W vector.
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = tensor_to_np_img(synth_image)
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    #moved the video to the end so we can look at images while these videos take time to compile
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for idx, projected_w in enumerate(projected_w_steps):
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = tensor_to_np_img(synth_image)
            if idx == 0:
                first_synth_image = synth_image
            video.append_data(np.concatenate([first_synth_image, synth_image], axis=1))
        video.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
