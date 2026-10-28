#!/usr/bin/env python3
"""Compare CPU and CUDA BACKWARD::render on one full bicycle SfM scene."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diff_gaussian_rasterization import get_last_backward_render_comparison
from fused_ssim import fused_ssim
from gaussian_renderer import render
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from utils.loss_utils import l1_loss


def make_pipeline():
    return SimpleNamespace(
        separate_sh=True,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        cpu_backward_render=True,
        compare_cpu_cuda_backward_render=True,
        debug=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/bicycle")
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="results/bicycle_sfm_backward_render_comparison.pt",
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
    package = render(camera, gaussians, make_pipeline(), background)
    image = package["render"]
    target = camera.original_image
    l1 = l1_loss(image, target)
    ssim_value = fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ssim_value)

    # This single autograd call executes CUDA and CPU BACKWARD::render from
    # the exact same saved forward buffers. The CPU result is propagated to
    # the model parameters; no optimizer step is performed.
    loss.backward()

    comparison = get_last_backward_render_comparison()
    if comparison is None:
        raise RuntimeError("BACKWARD::render comparison results were not produced")

    metadata = {
        "source": str(Path(args.source).resolve()),
        "images": args.images,
        "camera_index": args.camera_index,
        "camera_name": camera.image_name,
        "image_width": camera.image_width,
        "image_height": camera.image_height,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "loss": float(loss.detach().cpu()),
    }

    print(
        f"camera={metadata['camera_name']} "
        f"size={metadata['image_width']}x{metadata['image_height']} "
        f"gaussians={metadata['gaussian_count']} loss={metadata['loss']:.9g}"
    )
    for name, values in comparison.items():
        print(
            f"{name:16s} exact_equal={values['exact_equal']} "
            f"different={values['different_elements']}/{values['cpu'].numel()} "
            f"max_abs={values['max_absolute_difference']:.9g} "
            f"mean_abs={values['mean_absolute_difference']:.9g}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "backward_render": comparison,
        },
        output_path,
    )
    print(f"saved comparison tensors: {output_path}")


if __name__ == "__main__":
    main()
