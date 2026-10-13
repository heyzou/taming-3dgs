#!/usr/bin/env python3
"""Combine all budget timing charts into a readable comparison sheet."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SCENES = [
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
]
OUTPUT = HERE / "timing_100iter_stacked_bar_all.png"


def font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    source_size = (3600, 1440)
    tile_size = (1760, 704)
    gap = 24
    label_width = 220
    header_height = 120
    width = label_width + gap + tile_size[0] * 2 + gap
    height = header_height + len(SCENES) * tile_size[1] + (len(SCENES) - 1) * gap

    sheet = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(sheet)
    title_font = font(42, bold=True)
    column_font = font(32, bold=True)

    title = "Timing per 100 iterations — all budget benchmarks"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 20), title, fill="#111827", font=title_font)
    chart_x = label_width + gap
    draw.text((chart_x + tile_size[0] / 2 - 90, 76), "RTX 4090", fill="#374151", font=column_font)
    draw.text((chart_x + tile_size[0] + gap + tile_size[0] / 2 - 55, 76), "Jetson", fill="#374151", font=column_font)
    scene_font = font(34, bold=True)

    for row, scene in enumerate(SCENES):
        y = header_height + row * (tile_size[1] + gap)
        label = scene.capitalize()
        label_box = draw.textbbox((0, 0), label, font=scene_font)
        label_x = (label_width - (label_box[2] - label_box[0])) / 2
        label_y = y + (tile_size[1] - (label_box[3] - label_box[1])) / 2
        draw.text((label_x, label_y), label, fill="#111827", font=scene_font)
        for col, suffix in enumerate(("_budget", "_budget_jetson")):
            source = HERE / f"{scene}{suffix}" / "timing_100iter_stacked_bar.png"
            if not source.exists():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                image = image.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
                x = chart_x + col * (tile_size[0] + gap)
                sheet.paste(image, (x, y))

    sheet.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
