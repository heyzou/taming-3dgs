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
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/bicycle_cpu_backward_render_profile.csv",
    )
    parser.add_argument(
        "--output",
        default="results/bicycle_cpu_backward_render_profile.png",
    )
    parser.add_argument(
        "--nsys-breakdown",
        default="results/bicycle_cpu_backward_render_sections.csv",
    )
    args = parser.parse_args()

    with Path(args.input).open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    required = {"render_cpu_ms", "total_ms"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"CSV is missing columns: {sorted(missing)}")

    with Path(args.nsys_breakdown).open(newline="") as csv_file:
        nsys_rows = list(csv.DictReader(csv_file))

    values = {
        key: median(rows, key)
        for key in (
            "d2h_ms",
            "host_init_ms",
            "render_cpu_ms",
            "h2d_ms",
            "total_ms",
        )
    }
    image = Image.new("RGB", (1800, 1850), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "CPU BACKWARD::render profile", font=font(48, True), fill=COLORS["ink"])
    draw.text(
        (70, 105),
        "bicycle / images_8 / one camera / all 54,275 SfM Gaussians",
        font=font(25),
        fill=COLORS["gray"],
    )

    top_sections = sorted(
        nsys_rows, key=lambda row: float(row["estimated_ms"]), reverse=True
    )[:2]
    cards = [
        ("Total", values["total_ms"], COLORS["blue"]),
        ("render_cpu", values["render_cpu_ms"], COLORS["red"]),
        (top_sections[0]["section"], float(top_sections[0]["estimated_ms"]), COLORS["orange"]),
        (top_sections[1]["section"], float(top_sections[1]["estimated_ms"]), COLORS["cyan"]),
    ]
    for index, (label, value, color) in enumerate(cards):
        x0 = 70 + index * 425
        draw.rounded_rectangle((x0, 165, x0 + 390, 310), radius=18, fill="white", outline=COLORS["light"], width=2)
        draw.rectangle((x0, 165, x0 + 10, 310), fill=color)
        draw.text((x0 + 35, 190), label, font=font(20, True), fill=COLORS["gray"])
        draw.text((x0 + 35, 235), f"{value:.2f} ms", font=font(39, True), fill=COLORS["ink"])

    draw.text((70, 355), "Total backward breakdown", font=font(30, True), fill=COLORS["ink"])
    overall = [
        ("GPU to CPU", values["d2h_ms"], COLORS["blue"]),
        ("Host init", values["host_init_ms"], COLORS["green"]),
        ("render_cpu", values["render_cpu_ms"], COLORS["red"]),
        ("CPU to GPU", values["h2d_ms"], COLORS["cyan"]),
    ]
    bar_x, bar_y, bar_w, bar_h = 70, 410, 1660, 72
    cursor = bar_x
    for label, value, color in overall:
        width = max(3, bar_w * value / values["total_ms"])
        draw.rectangle((cursor, bar_y, min(bar_x + bar_w, cursor + width), bar_y + bar_h), fill=color)
        cursor += width
    legend_x = 70
    for label, value, color in overall:
        draw.rectangle((legend_x, 505, legend_x + 24, 529), fill=color)
        percentage = value / values["total_ms"] * 100.0
        draw.text((legend_x + 34, 501), f"{label}: {value:.2f} ms ({percentage:.2f}%)", font=font(21), fill=COLORS["ink"])
        legend_x += 415

    draw.text((70, 585), "Inside render_cpu - Nsight CPU sampling", font=font(30, True), fill=COLORS["ink"])
    section_colors = {
        "Pixel eligibility check": COLORS["cyan"],
        "Gaussian exponent": COLORS["orange"],
        "Gaussian weight": "#8b5cf6",
        "Alpha calculation": "#fb7185",
        "Alpha threshold": COLORS["red"],
        "Color gradient": COLORS["blue"],
        "Foreground alpha gradient": "#0ea5e9",
        "Background alpha gradient": "#6366f1",
        "Mean gradient": COLORS["green"],
        "Conic / opacity gradient": "#84cc16",
    }
    max_percent = max(float(row["share_percent"]) for row in nsys_rows)
    for index, row in enumerate(nsys_rows):
        label = row["section"]
        samples = int(row["samples"])
        percentage = float(row["share_percent"])
        color = section_colors.get(label, COLORS["gray"])
        y = 645 + index * 62
        draw.text((70, y + 5), label, font=font(20, True), fill=COLORS["ink"])
        draw.rounded_rectangle((430, y, 1430, y + 43), radius=8, fill=COLORS["light"])
        width = 1000 * percentage / max_percent
        draw.rounded_rectangle((430, y, 430 + width, y + 43), radius=8, fill=color)
        draw.text((1460, y + 5), f"{percentage:.2f}%  ({samples:,})", font=font(19, True), fill=COLORS["ink"])

    draw.text((70, 1540), "Workload counters", font=font(30, True), fill=COLORS["ink"])
    counters = [
        ("Gaussian x pixel tests", "gaussian_pixel_tests"),
        ("Passed pixel checks", "pixel_checks_passed"),
        ("Power evaluations", "power_evaluations"),
        ("Gradient evaluations", "gradient_evaluations"),
    ]
    for index, (label, key) in enumerate(counters):
        x = 70 + index * 425
        value = int(rows[0][key])
        draw.text((x, 1595), label, font=font(20, True), fill=COLORS["gray"])
        draw.text((x, 1635), f"{value:,}", font=font(29, True), fill=COLORS["ink"])

    render_runs = [float(row["render_cpu_ms"]) for row in rows]
    draw.text(
        (70, 1760),
        f"Median of {len(rows)} measured runs (warm-up excluded). "
        f"render_cpu range: {min(render_runs):.2f}-{max(render_runs):.2f} ms. "
        "Section times are estimated from Nsight sample share x render_cpu median.",
        font=font(20),
        fill=COLORS["gray"],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
