#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "ink": "#172033",
    "gray": "#64748b",
    "line": "#cbd5e1",
    "header": "#1e3a8a",
    "header_text": "#ffffff",
    "row": "#ffffff",
    "alternate": "#f1f5f9",
    "positive": "#166534",
    "zero": "#c2410c",
    "note": "#fff7ed",
}

SECTION_NAMES = {
    "gaussian_weight": "Gaussian weight",
    "tile_pixel_preparation": "Tile / pixel preparation",
    "output_accumulation": "Output accumulation",
    "gaussian_preparation": "Gaussian preparation",
    "pixel_eligibility_check": "Pixel eligibility check",
    "gaussian_exponent": "Gaussian exponent",
    "alpha_calculation": "Alpha calculation",
    "alpha_threshold": "Alpha threshold",
    "color_gradient": "Color gradient",
    "foreground_alpha_gradient": "Foreground alpha gradient",
    "background_alpha_gradient": "Background alpha gradient",
    "transmittance_update": "Transmittance update",
    "mean_gradient": "Mean gradient",
    "conic_opacity_gradient": "Conic / opacity gradient",
}


def font(size, bold=False):
    name = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(f"/usr/share/fonts/opentype/noto/{name}", size, index=2)


LOGICAL_CALLS = {
    "tile_pixel_preparation": ("buckets", "bucket"),
    "gaussian_preparation": ("splats", "Gaussian splat"),
    "pixel_eligibility_check": ("gaussian_pixel_tests", "Gaussian x pixel candidate"),
    "gaussian_exponent": ("power_evaluations", "power evaluation"),
    "gaussian_weight": ("exp_evaluations", "exp evaluation"),
    "alpha_calculation": ("exp_evaluations", "alpha evaluation"),
    "alpha_threshold": ("exp_evaluations", "alpha threshold evaluation"),
    "color_gradient": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "foreground_alpha_gradient": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "background_alpha_gradient": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "transmittance_update": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "mean_gradient": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "conic_opacity_gradient": ("gradient_evaluations", "gradient Gaussian x pixel"),
    "output_accumulation": ("splats", "Gaussian splat"),
}


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


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
        "--csv-output",
        default="results/bicycle_cpu_backward_render_timing_table.csv",
    )
    parser.add_argument(
        "--image-output",
        default="results/bicycle_cpu_backward_render_timing_table.png",
    )
    args = parser.parse_args()

    with Path(args.sections).open(newline="") as csv_file:
        sections = list(csv.DictReader(csv_file))
    with Path(args.profile).open(newline="") as csv_file:
        profiles = list(csv.DictReader(csv_file))
    if not sections or not profiles:
        raise RuntimeError("Timing CSV is empty")

    rows = []
    for section in sections:
        name = section["section"]
        counter_key, logical_unit = LOGICAL_CALLS[name]
        logical_calls = int(median(profiles, counter_key))
        timer_intervals = int(float(section["intervals_median"]))
        zero_tick_intervals = int(float(section["zero_tick_intervals_median"]))
        raw_ms = float(section["raw_ms_median"])
        overhead_ms = float(section["timer_overhead_ms_median"])
        corrected_ms = float(section["corrected_ms_median"])
        rows.append(
            {
                "rank": int(section["rank"]),
                "section": name,
                "logical_unit": logical_unit,
                "logical_calls": logical_calls,
                "timer_intervals": timer_intervals,
                "zero_tick_intervals": zero_tick_intervals,
                "raw_ms": raw_ms,
                "overhead_subtracted_ms": overhead_ms,
                "corrected_ms": corrected_ms,
                "raw_ns_per_logical_call": raw_ms * 1.0e6 / logical_calls,
                "overhead_ns_per_logical_call": overhead_ms * 1.0e6 / logical_calls,
                "corrected_ns_per_logical_call": corrected_ms * 1.0e6 / logical_calls,
                "raw_ns_per_timer_interval": raw_ms * 1.0e6 / timer_intervals,
                "corrected_ns_per_timer_interval": corrected_ms * 1.0e6 / timer_intervals,
                "corrected_share_percent": float(section["share_percent"]),
            }
        )

    fieldnames = list(rows[0])
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    image_width = 3335
    table_left = 55
    table_right = image_width - table_left
    image = Image.new("RGB", (image_width, 1950), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 35),
        "CPU BACKWARD::render 直接計測詳細表",
        font=font(60, True),
        fill=COLORS["ink"],
    )
    draw.text(
        (55, 112),
        "bicycle / images_8 / 5回の中央値 / ARM cntvct_el0",
        font=font(32),
        fill=COLORS["gray"],
    )
    draw.rounded_rectangle(
        (table_left, 163, table_right, 272),
        radius=12,
        fill=COLORS["note"],
        outline="#fb923c",
        width=2,
    )
    draw.text(
        (75, 178),
        "0 tick回数は生の計測差が0だった回数です。",
        font=font(29, True),
        fill="#9a3412",
    )
    draw.text(
        (75, 228),
        "0.000*は推定オーバーヘッドが実測値以上になった値であり、処理時間が0という意味ではありません。",
        font=font(27),
        fill="#9a3412",
    )

    column_specs = [
        ("順位", 100, "center"),
        ("区間", 500, "left"),
        ("論理\n呼出回数", 320, "right"),
        ("タイマー\n計測回数", 320, "right"),
        ("0 tick\n計測回数", 300, "right"),
        ("実測時間\nms", 235, "right"),
        ("差引オーバーヘッド\nms", 320, "right"),
        ("補正後時間\nms", 250, "right"),
        ("補正前 ns /\n論理呼出", 340, "right"),
        ("補正後 ns /\n論理呼出", 340, "right"),
        ("補正後\n割合 %", 200, "right"),
    ]
    columns = []
    column_x = table_left
    for label, width, align in column_specs:
        columns.append((label, column_x, width, align))
        column_x += width
    header_y = 300
    header_height = 104
    row_height = 100
    draw.rectangle((table_left, header_y, table_right, header_y + header_height), fill=COLORS["header"])
    for label, x, width, align in columns:
        box = draw.multiline_textbbox((0, 0), label, font=font(27, True), spacing=4)
        text_width = box[2] - box[0]
        if align == "left":
            text_x = x + 12
        elif align == "center":
            text_x = x + (width - text_width) / 2
        else:
            text_x = x + width - text_width - 12
        draw.multiline_text(
            (text_x, header_y + 19),
            label,
            font=font(27, True),
            fill=COLORS["header_text"],
            spacing=4,
            align="right" if align == "right" else align,
        )

    for index, row in enumerate(rows):
        y = header_y + header_height + row_height * index
        fill = COLORS["row"] if index % 2 == 0 else COLORS["alternate"]
        draw.rectangle((table_left, y, table_right, y + row_height), fill=fill)
        values = [
            str(row["rank"]),
            SECTION_NAMES[row["section"]],
            f"{row['logical_calls']:,}",
            f"{row['timer_intervals']:,}",
            f"{row['zero_tick_intervals']:,}",
            f"{row['raw_ms']:.3f}",
            f"{row['overhead_subtracted_ms']:.3f}",
            f"{row['corrected_ms']:.3f}" if row["corrected_ms"] else "0.000*",
            f"{row['raw_ns_per_logical_call']:.3f}",
            f"{row['corrected_ns_per_logical_call']:.3f}" if row["corrected_ms"] else "0.000*",
            f"{row['corrected_share_percent']:.2f}",
        ]
        for value, (_, x, width, align) in zip(values, columns):
            text_box = draw.textbbox((0, 0), value, font=font(28, align == "left"))
            text_width = text_box[2] - text_box[0]
            if align == "left":
                text_x = x + 12
            elif align == "center":
                text_x = x + (width - text_width) / 2
            else:
                text_x = x + width - text_width - 12
            value_color = COLORS["ink"]
            if value == "0.000*":
                value_color = COLORS["zero"]
            elif align == "left" and row["corrected_ms"] > 0:
                value_color = COLORS["positive"]
            draw.text(
                (text_x, y + 33),
                value,
                font=font(28, align == "left"),
                fill=value_color,
            )
        draw.line((table_left, y + row_height, table_right, y + row_height), fill=COLORS["line"], width=1)

    table_bottom = header_y + header_height + row_height * len(rows)
    draw.text(
        (55, table_bottom + 30),
        "表の回数は14区間本体のみ。各計測runでは校正用に2,001,000回のタイマー計測を追加します。",
        font=font(27),
        fill=COLORS["gray"],
    )
    draw.text(
        (55, table_bottom + 66),
        "補正後時間 = max(0, 実測時間 - 校正済み1回当たり負荷 × タイマー計測回数)",
        font=font(27),
        fill=COLORS["gray"],
    )

    image_output = Path(args.image_output)
    image_output.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_output)

    print(csv_output)
    print(image_output)


if __name__ == "__main__":
    main()
