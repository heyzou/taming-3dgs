#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "blue": "#2563eb",
    "cyan": "#06b6d4",
    "orange": "#f59e0b",
    "red": "#ef4444",
    "green": "#22c55e",
    "gray": "#64748b",
    "light": "#e2e8f0",
    "ink": "#172033",
    "warning_bg": "#fff7ed",
    "warning_border": "#fb923c",
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def display_name(name):
    return name.replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        default="results/bicycle_cpu_backward_render_sections_direct.csv",
    )
    parser.add_argument(
        "--profile",
        default="results/bicycle_cpu_backward_render_profile.csv",
    )
    parser.add_argument(
        "--output",
        default="results/bicycle_cpu_backward_render_direct_profile.png",
    )
    args = parser.parse_args()

    with Path(args.sections).open(newline="") as csv_file:
        sections = list(csv.DictReader(csv_file))
    with Path(args.profile).open(newline="") as csv_file:
        profiles = list(csv.DictReader(csv_file))
    if not sections or not profiles:
        raise RuntimeError("Direct timing CSV is empty")

    render_ms = median(profiles, "render_cpu_ms")
    timer_frequency_hz = median(profiles, "timer_frequency_hz")
    timer_overhead_ticks = median(profiles, "timer_overhead_ticks")
    tick_ns = 1.0e9 / timer_frequency_hz
    section_total_ms = sum(float(row["corrected_ms_median"]) for row in sections)
    max_ms = max(float(row["corrected_ms_median"]) for row in sections)

    image = Image.new("RGB", (1900, 1720), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42),
        "C++ Direct Section Timing Profile",
        font=font(48, True),
        fill=COLORS["ink"],
    )
    draw.text(
        (70, 104),
        "bicycle / images_8 / ARM cntvct_el0 / median of 5 measured runs",
        font=font(25),
        fill=COLORS["gray"],
    )

    cards = [
        ("Instrumented render_cpu", render_ms, "ms", COLORS["red"]),
        ("14-section sum", section_total_ms, "ms", COLORS["blue"]),
        ("Counter resolution", tick_ns, "ns/tick", COLORS["orange"]),
        ("Calibrated overhead", timer_overhead_ticks, "ticks", COLORS["cyan"]),
    ]
    for index, (label, value, unit, color) in enumerate(cards):
        x0 = 70 + index * 455
        draw.rounded_rectangle(
            (x0, 165, x0 + 420, 310),
            radius=18,
            fill="white",
            outline=COLORS["light"],
            width=2,
        )
        draw.rectangle((x0, 165, x0 + 10, 310), fill=color)
        draw.text((x0 + 35, 190), label, font=font(20, True), fill=COLORS["gray"])
        draw.text(
            (x0 + 35, 235),
            f"{value:.2f} {unit}",
            font=font(34, True),
            fill=COLORS["ink"],
        )

    warning = timer_overhead_ticks == 0
    if warning:
        draw.rounded_rectangle(
            (70, 340, 1830, 445),
            radius=14,
            fill=COLORS["warning_bg"],
            outline=COLORS["warning_border"],
            width=3,
        )
        draw.text(
            (95, 360),
            "Measurement warning: timer overhead was calibrated as 0 ticks.",
            font=font(24, True),
            fill="#9a3412",
        )
        draw.text(
            (95, 400),
            "Sections near one 32 ns counter tick are strongly affected by timer resolution; use their ranking cautiously.",
            font=font(21),
            fill="#9a3412",
        )

    draw.text(
        (70, 485),
        "Corrected median by section",
        font=font(31, True),
        fill=COLORS["ink"],
    )
    draw.text(
        (1440, 490),
        "time / share / ns per interval",
        font=font(19, True),
        fill=COLORS["gray"],
    )

    palette = [
        COLORS["red"], COLORS["blue"], COLORS["orange"], COLORS["cyan"],
        "#8b5cf6", "#f97316", "#fb7185", COLORS["green"],
        "#6366f1", "#84cc16", "#0ea5e9", "#64748b", "#94a3b8", "#475569",
    ]
    for index, row in enumerate(sections):
        label = display_name(row["section"])
        value = float(row["corrected_ms_median"])
        share = float(row["share_percent"])
        ns_per_interval = float(row["corrected_ns_per_interval_median"])
        y = 545 + index * 76
        draw.text((70, y + 6), f"{index + 1:2d}. {label}", font=font(19, True), fill=COLORS["ink"])
        draw.rounded_rectangle((570, y, 1390, y + 46), radius=8, fill=COLORS["light"])
        width = 820 * value / max_ms
        draw.rounded_rectangle(
            (570, y, 570 + max(3, width), y + 46),
            radius=8,
            fill=palette[index],
        )
        draw.text(
            (1420, y + 5),
            f"{value:8.2f} ms   {share:5.2f}%   {ns_per_interval:7.2f} ns",
            font=font(18, True),
            fill=COLORS["ink"],
        )

    draw.text(
        (70, 1655),
        "Percentages use the sum of the 14 reported corrected medians, not the instrumented render_cpu wall time.",
        font=font(20),
        fill=COLORS["gray"],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
