#!/usr/bin/env python3
"""Plot overall timing and backward timing breakdown from benchmark arrays."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    if values.size % block_size:
        raise ValueError(f"array length {values.size} is not divisible by {block_size}")
    return values.reshape(-1, block_size).mean(axis=1)


def plot(directory: Path, block_size: int) -> Path:
    forward = np.load(directory / "forward.npy")
    backward = np.load(directory / "backward.npy")
    ssim = np.load(directory / "ssim_backward.npy")
    rasterizer = np.load(directory / "rasterizer_backward.npy")
    rasterizer_render = np.load(directory / "rasterizer_render_backward.npy")
    rasterizer_preprocess = np.load(directory / "rasterizer_preprocess_backward.npy")
    step = np.load(directory / "step.npy")

    arrays = (forward, backward, ssim, rasterizer, rasterizer_render, rasterizer_preprocess, step)
    if len({array.size for array in arrays}) != 1:
        raise ValueError("timing arrays have different lengths")

    # The residual contains L1 backward plus Autograd/gradient-accumulation work.
    residual = backward - ssim - rasterizer
    if np.any(residual < 0):
        raise ValueError("backward residual contains negative values")
    rasterizer_residual = rasterizer - rasterizer_render - rasterizer_preprocess
    if np.any(rasterizer_residual < 0):
        raise ValueError("rasterizer residual contains negative values")

    x = np.arange(block_mean(backward, block_size).size) * block_size + 1
    forward_m = block_mean(forward, block_size)
    backward_m = block_mean(backward, block_size)
    step_m = block_mean(step, block_size)
    ssim_m = block_mean(ssim, block_size)
    rasterizer_m = block_mean(rasterizer, block_size)
    rasterizer_render_m = block_mean(rasterizer_render, block_size)
    rasterizer_preprocess_m = block_mean(rasterizer_preprocess, block_size)
    rasterizer_residual_m = block_mean(rasterizer_residual, block_size)
    residual_m = block_mean(residual, block_size)

    fig, (overall_ax, breakdown_ax, rasterizer_ax) = plt.subplots(
        3, 1, figsize=(16, 14), sharex=True, constrained_layout=True
    )

    width = block_size * 0.9
    overall_ax.bar(x, forward_m, width=width, align="edge", label="Forward", color="#2f66e5")
    overall_ax.bar(
        x, backward_m, width=width, align="edge", bottom=forward_m,
        label="Backward", color="#e32226"
    )
    overall_ax.bar(
        x, step_m, width=width, align="edge", bottom=forward_m + backward_m,
        label="Optimizer step", color="#16a34a"
    )
    overall_ax.set_ylabel("Mean time per iteration (ms)")
    overall_ax.set_title("Overall training timing")
    overall_ax.grid(axis="y", alpha=0.25)
    overall_ax.set_axisbelow(True)
    overall_ax.legend(loc="upper left")

    breakdown_ax.bar(
        x, residual_m, width=width, align="edge",
        label="L1 + other backward", color="#f59e0b"
    )
    breakdown_ax.bar(
        x, ssim_m, width=width, align="edge", bottom=residual_m,
        label="SSIM backward", color="#8b5cf6"
    )
    breakdown_ax.bar(
        x, rasterizer_m, width=width, align="edge", bottom=residual_m + ssim_m,
        label="Rasterizer backward", color="#dc2626"
    )
    breakdown_ax.plot(x + width / 2, backward_m, color="#111827", linewidth=1.5, label="Backward total")
    breakdown_ax.set_xlabel(f"Iteration block (each bar = {block_size} iterations)")
    breakdown_ax.set_ylabel("Mean time per iteration (ms)")
    breakdown_ax.set_title("Backward timing breakdown")
    breakdown_ax.grid(axis="y", alpha=0.25)
    breakdown_ax.set_axisbelow(True)
    breakdown_ax.legend(loc="upper left")

    rasterizer_ax.bar(
        x, rasterizer_residual_m, width=width, align="edge",
        label="Rasterizer setup / other", color="#9ca3af"
    )
    rasterizer_ax.bar(
        x, rasterizer_render_m, width=width, align="edge",
        bottom=rasterizer_residual_m, label="BACKWARD::render", color="#ef4444"
    )
    rasterizer_ax.bar(
        x, rasterizer_preprocess_m, width=width, align="edge",
        bottom=rasterizer_residual_m + rasterizer_render_m,
        label="BACKWARD::preprocess", color="#2563eb"
    )
    rasterizer_ax.plot(
        x + width / 2, rasterizer_m, color="#111827", linewidth=1.5,
        label="Rasterizer backward total"
    )
    rasterizer_ax.set_xlabel(f"Iteration block (each bar = {block_size} iterations)")
    rasterizer_ax.set_ylabel("Mean time per iteration (ms)")
    rasterizer_ax.set_title("Gaussian Rasterizer backward breakdown")
    rasterizer_ax.grid(axis="y", alpha=0.25)
    rasterizer_ax.set_axisbelow(True)
    rasterizer_ax.legend(loc="upper left")

    output = directory / "backward_breakdown.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("eval/bicycle_budget_jetson/backward_breakdown"),
    )
    parser.add_argument("--block-size", type=int, default=100)
    args = parser.parse_args()
    print(plot(args.directory, args.block_size))


if __name__ == "__main__":
    main()
