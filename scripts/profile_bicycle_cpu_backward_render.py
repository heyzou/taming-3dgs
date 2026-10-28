#!/usr/bin/env python3
"""Profile sequential CPU BACKWARD::render on one full bicycle SfM scene."""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diff_gaussian_rasterization import get_last_cpu_backward_profile
from fused_ssim import fused_ssim
from gaussian_renderer import render
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from utils.loss_utils import l1_loss


def make_pipeline(profile):
    return SimpleNamespace(
        separate_sh=True,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        cpu_backward_render=True,
        compare_cpu_cuda_backward_render=False,
        profile_cpu_backward_render=profile,
        debug=False,
    )


def clear_gradients(gaussians):
    for parameter in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        parameter.grad = None


def run_backward(gaussians, camera, background, lambda_dssim, profile):
    clear_gradients(gaussians)
    package = render(camera, gaussians, make_pipeline(profile), background)
    image = package["render"]
    target = camera.original_image
    l1 = l1_loss(image, target)
    ssim_value = fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)
    loss.backward()
    torch.cuda.synchronize()
    return get_last_cpu_backward_profile() if profile else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/bicycle")
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=5)
    parser.add_argument("--sampling-runs", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/bicycle_cpu_backward_render_profile.csv",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    scene_info = readColmapSceneInfo(args.source, args.images, eval=True)
    if not 0 <= args.camera_index < len(scene_info.train_cameras):
        raise ValueError(f"camera-index must be below {len(scene_info.train_cameras)}")

    camera_args = SimpleNamespace(resolution=1, data_device="cuda")
    camera = loadCam(
        camera_args,
        args.camera_index,
        scene_info.train_cameras[args.camera_index],
        1.0,
    )
    gaussians = GaussianModel(3)
    gaussians.create_from_pcd(
        scene_info.point_cloud,
        scene_info.nerf_normalization["radius"],
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    for _ in range(args.warmup_runs):
        run_backward(gaussians, camera, background, args.lambda_dssim, False)

    profiles = []
    for run in range(1, args.profile_runs + 1):
        profile = run_backward(gaussians, camera, background, args.lambda_dssim, True)
        profile["run"] = run
        profiles.append(profile)

    # These repetitions intentionally disable timers and counters. Launch this
    # mode under Nsight Systems to sample optimized render_cpu source lines.
    for _ in range(args.sampling_runs):
        run_backward(gaussians, camera, background, args.lambda_dssim, False)

    if profiles:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run"] + [key for key in profiles[0] if key != "run"]
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(profiles)

        print(
            f"camera={camera.image_name} size={camera.image_width}x{camera.image_height} "
            f"gaussians={gaussians.get_xyz.shape[0]}"
        )
        for key in (
            "d2h_ms",
            "host_init_ms",
            "render_cpu_ms",
            "h2d_ms",
            "total_ms",
        ):
            values = torch.tensor([profile[key] for profile in profiles])
            print(f"{key:20s} median={values.median().item():.3f} ms")
        print(f"profile CSV: {output_path}")


if __name__ == "__main__":
    main()
