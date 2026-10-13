#!/usr/bin/env python3
"""Create 100-iteration stacked timing bars for budget benchmark outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BLOCK_SIZE = 100

CAD_MACHINE = {
    "bicycle": "cad101",
    "bonsai": "cad102",
    "counter": "cad101",
    "flowers": "cad102",
    "garden": "cad101",
    "kitchen": "cad102",
    "room": "cad101",
    "stump": "cad102",
    "treehill": "cad101",
}

WONDER_MACHINE = {
    "bicycle": "wonder44",
    "bonsai": "wonder45",
    "counter": "wonder44",
    "flowers": "wonder45",
    "garden": "wonder44",
    "kitchen": "wonder45",
    "room": "wonder44",
    "stump": "wonder45",
    "treehill": "wonder44",
}


def plot_benchmark(directory: Path) -> None:
    forward = np.load(directory / "forward.npy")
    backward = np.load(directory / "backward.npy")
    step = np.load(directory / "step.npy")

    if not (len(forward) == len(backward) == len(step)):
        raise ValueError(f"timing arrays have different lengths in {directory}")
    if len(forward) % BLOCK_SIZE:
        raise ValueError(f"timing arrays are not a multiple of {BLOCK_SIZE}: {directory}")

    def block_sums(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, BLOCK_SIZE).sum(axis=1)

    forward_blocks = block_sums(forward)
    backward_blocks = block_sums(backward)
    step_blocks = block_sums(step)
    x = np.arange(len(forward_blocks)) * BLOCK_SIZE

    is_jetson = directory.name.endswith("_budget_jetson")
    scene = directory.name.removesuffix("_budget_jetson" if is_jetson else "_budget")
    platform = "Jetson" if is_jetson else "RTX 4090"
    machine = (WONDER_MACHINE if is_jetson else CAD_MACHINE)[scene]
    title = f"{scene.capitalize()} budget: total processing time per 100 iterations ({platform}, {machine})"

    fig, ax = plt.subplots(figsize=(36, 14.4), dpi=100)
    ax.bar(x, forward_blocks, width=BLOCK_SIZE, align="edge", color="#2f66e5", label="Forward")
    ax.bar(
        x,
        backward_blocks,
        width=BLOCK_SIZE,
        align="edge",
        bottom=forward_blocks,
        color="#e32226",
        label="Backward",
    )
    ax.bar(
        x,
        step_blocks,
        width=BLOCK_SIZE,
        align="edge",
        bottom=forward_blocks + backward_blocks,
        color="#16a34a",
        label="Optimizer step",
    )
    ax.set_xlabel("Iteration block (each bar = iterations 1-100, 101-200, ...)")
    ax.set_ylabel("Total time per 100 iterations (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(directory / "timing_100iter_stacked_bar.png")
    plt.close(fig)


def main() -> None:
    for directory in sorted(HERE.glob("*_budget*")):
        if all((directory / name).exists() for name in ("forward.npy", "backward.npy", "step.npy")):
            plot_benchmark(directory)
            print(directory / "timing_100iter_stacked_bar.png")


if __name__ == "__main__":
    main()
