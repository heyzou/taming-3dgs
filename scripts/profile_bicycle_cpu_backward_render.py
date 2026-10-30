#!/usr/bin/env python3
"""Profile sequential CPU BACKWARD::render on one bicycle SfM image."""

import argparse
import csv
import statistics
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
    parser.add_argument(
        "--section-output",
        default="results/bicycle_cpu_backward_render_sections_direct.csv",
    )
    parser.add_argument(
        "--zero-tick-output",
        default="results/bicycle_cpu_backward_render_zero_ticks.csv",
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
        fieldnames = [
            "run",
            *[key for key in profiles[0] if key not in ("run", "sections")],
        ]
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                {key: profile[key] for key in fieldnames} for profile in profiles
            )

        section_runs = {}
        zero_tick_rows = []
        for profile in profiles:
            for section in profile["sections"]:
                section_runs.setdefault(section["name"], []).append(section)
                intervals = section["intervals"]
                zero_ticks = section["zero_tick_intervals"]
                zero_tick_rows.append(
                    {
                        "run": profile["run"],
                        "section": section["name"],
                        "intervals": intervals,
                        "zero_tick_intervals": zero_ticks,
                        "zero_tick_percent":
                            100.0 * zero_ticks / intervals if intervals else 0.0,
                    }
                )

        zero_tick_output_path = Path(args.zero_tick_output)
        zero_tick_output_path.parent.mkdir(parents=True, exist_ok=True)
        with zero_tick_output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=(
                    "run",
                    "section",
                    "intervals",
                    "zero_tick_intervals",
                    "zero_tick_percent",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(zero_tick_rows)

        section_rows = []
        for name, runs in section_runs.items():
            section_rows.append(
                {
                    "section": name,
                    "corrected_ms_median": statistics.median(
                        run["corrected_ms"] for run in runs
                    ),
                    "raw_ms_median": statistics.median(
                        run["raw_ms"] for run in runs
                    ),
                    "timer_overhead_ms_median": statistics.median(
                        run["timer_overhead_ms"] for run in runs
                    ),
                    "intervals_median": statistics.median(
                        run["intervals"] for run in runs
                    ),
                    "zero_tick_intervals_median": statistics.median(
                        run["zero_tick_intervals"] for run in runs
                    ),
                    "corrected_ns_per_interval_median": statistics.median(
                        run["corrected_ns_per_interval"] for run in runs
                    ),
                }
            )
        section_rows.sort(key=lambda row: row["corrected_ms_median"], reverse=True)
        corrected_total = sum(row["corrected_ms_median"] for row in section_rows)
        for rank, row in enumerate(section_rows, start=1):
            row["rank"] = rank
            row["share_percent"] = (
                100.0 * row["corrected_ms_median"] / corrected_total
                if corrected_total
                else 0.0
            )

        section_output_path = Path(args.section_output)
        section_output_path.parent.mkdir(parents=True, exist_ok=True)
        section_fieldnames = [
            "rank",
            "section",
            "corrected_ms_median",
            "share_percent",
            "raw_ms_median",
            "timer_overhead_ms_median",
            "intervals_median",
            "zero_tick_intervals_median",
            "corrected_ns_per_interval_median",
        ]
        with section_output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=section_fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(section_rows)

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
        timer_frequency_hz = statistics.median(
            profile["timer_frequency_hz"] for profile in profiles
        )
        timer_overhead_ticks = statistics.median(
            profile["timer_overhead_ticks"] for profile in profiles
        )
        timer_full_overhead_ticks = statistics.median(
            profile["timer_full_overhead_ticks"] for profile in profiles
        )
        print(
            "timer interval overhead "
            f"median={timer_overhead_ticks:.6f} ticks "
            f"({timer_overhead_ticks * 1.0e9 / timer_frequency_hz:.3f} ns)"
        )
        print(
            "timer full overhead     "
            f"median={timer_full_overhead_ticks:.6f} ticks "
            f"({timer_full_overhead_ticks * 1.0e9 / timer_frequency_hz:.3f} ns)"
        )
        print(f"profile CSV: {output_path}")
        print("direct section timing (corrected median):")
        for row in section_rows:
            print(
                f"{row['rank']:2d}. {row['section']:32s} "
                f"{row['corrected_ms_median']:10.3f} ms "
                f"({row['share_percent']:6.2f}%) "
                f"zero ticks={row['zero_tick_intervals_median']:,.0f}"
            )
        print(f"direct section CSV: {section_output_path}")
        print(f"zero tick counts CSV: {zero_tick_output_path}")


if __name__ == "__main__":
    main()
