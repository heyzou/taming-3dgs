#!/usr/bin/env python3

import argparse
import csv
import re
import sqlite3
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


SECTIONS = [
    "Tile / pixel preparation",
    "Gaussian preparation",
    "Pixel eligibility check",
    "Gaussian exponent",
    "Gaussian weight",
    "Alpha calculation",
    "Alpha threshold",
    "Color gradient",
    "Foreground alpha gradient",
    "Background alpha gradient",
    "Transmittance update",
    "Mean gradient",
    "Conic / opacity gradient",
    "Output accumulation",
]

COUNT_KEYS = {
    "Tile / pixel preparation": "buckets",
    "Gaussian preparation": "splats",
    "Pixel eligibility check": "gaussian_pixel_tests",
    "Gaussian exponent": "power_evaluations",
    "Gaussian weight": "exp_evaluations",
    "Alpha calculation": "exp_evaluations",
    "Alpha threshold": "exp_evaluations",
    "Color gradient": "gradient_evaluations",
    "Foreground alpha gradient": "gradient_evaluations",
    "Background alpha gradient": "gradient_evaluations",
    "Transmittance update": "gradient_evaluations",
    "Mean gradient": "gradient_evaluations",
    "Conic / opacity gradient": "gradient_evaluations",
    "Output accumulation": "splats",
}


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def read_profile(path):
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f"No profile rows in {path}")
    render_ms = statistics.median(float(row["render_cpu_ms"]) for row in rows)
    counts = {
        key: int(statistics.median(int(row[key]) for row in rows))
        for key in set(COUNT_KEYS.values())
    }
    return render_ms, counts


def read_markers(source_path):
    markers = []
    pattern = re.compile(r"PROFILE_SECTION:\s*(.+?)\s*$")
    for line_number, line in enumerate(source_path.read_text().splitlines(), 1):
        match = pattern.search(line)
        if match:
            markers.append((line_number, match.group(1)))
    found = {name for _, name in markers}
    missing = set(SECTIONS).difference(found)
    if missing:
        raise RuntimeError(f"Missing PROFILE_SECTION markers: {sorted(missing)}")
    return markers


def read_render_symbol(extension):
    for line in run(["nm", "-C", "-S", str(extension)]).splitlines():
        if " BACKWARD::render_cpu(" not in line:
            continue
        fields = line.split(maxsplit=3)
        return int(fields[0], 16), int(fields[1], 16)
    raise RuntimeError("BACKWARD::render_cpu symbol was not found")


def source_line(extension, runtime_ip, base, source_name, cache):
    key = (runtime_ip, base)
    if key in cache:
        return cache[key]
    relative_ip = runtime_ip - base
    output = run(
        ["addr2line", "-e", str(extension), "-f", "-C", "-i", hex(relative_ip)]
    )
    matches = re.findall(rf"{re.escape(source_name)}:(\d+)", output)
    cache[key] = int(matches[-1]) if matches else None
    return cache[key]


def find_load_base(extension, symbol_address, symbol_size, weighted_ips, source_name):
    minimum_ip = min(weighted_ips)
    low = minimum_ip - symbol_address - symbol_size
    high = minimum_ip - symbol_address
    alignment = 0x1000
    first = (low + alignment - 1) // alignment * alignment
    candidates = list(range(first, high + 1, alignment))
    if not candidates:
        raise RuntimeError("Could not derive the extension load address")

    cache = {}
    test_ips = sorted(weighted_ips, key=weighted_ips.get, reverse=True)[:80]
    scored = []
    for base in candidates:
        score = sum(
            weighted_ips[ip]
            for ip in test_ips
            if source_line(extension, ip, base, source_name, cache) is not None
        )
        scored.append((score, base))
    score, base = max(scored)
    if score == 0:
        raise RuntimeError("Could not map sampled addresses to backward_cpu.cpp")
    return base, cache


def section_for_line(line_number, markers):
    section = SECTIONS[0]
    for marker_line, marker_name in markers:
        if marker_line > line_number:
            break
        section = marker_name
    return section


def analyze(sqlite_path, extension, source_path, markers):
    connection = sqlite3.connect(sqlite_path)
    render_pattern = "BACKWARD::render_cpu(%"
    query = """
        SELECT cc.originalIP, COUNT(*)
        FROM SAMPLING_CALLCHAINS cc
        JOIN StringIds symbol ON symbol.id = cc.symbol
        WHERE symbol.value LIKE ?
        GROUP BY cc.originalIP
    """
    weighted_ips = Counter(dict(connection.execute(query, (render_pattern,))))
    if not weighted_ips:
        raise RuntimeError("No BACKWARD::render_cpu samples were found")

    symbol_address, symbol_size = read_render_symbol(extension)
    base, line_cache = find_load_base(
        extension,
        symbol_address,
        symbol_size,
        weighted_ips,
        source_path.name,
    )

    query = """
        WITH render_ids AS (
            SELECT DISTINCT cc.id
            FROM SAMPLING_CALLCHAINS cc
            JOIN StringIds symbol ON symbol.id = cc.symbol
            WHERE symbol.value LIKE ?
        )
        SELECT cc.id, cc.stackDepth, cc.originalIP, symbol.value, module.value
        FROM render_ids ids
        JOIN SAMPLING_CALLCHAINS cc ON cc.id = ids.id
        JOIN StringIds symbol ON symbol.id = cc.symbol
        JOIN StringIds module ON module.id = cc.module
        ORDER BY cc.id, cc.stackDepth
    """
    frames_by_sample = defaultdict(list)
    for sample_id, depth, ip, symbol, module in connection.execute(
        query, (render_pattern,)
    ):
        frames_by_sample[sample_id].append((depth, ip, symbol, module))

    extension_resolved = extension.resolve()
    samples = Counter()
    unclassified = 0
    for frames in frames_by_sample.values():
        leaf_symbol = frames[0][2]
        if any(token in leaf_symbol for token in ("expf", "xflowf", "with_errnof")):
            samples["Gaussian weight"] += 1
            continue

        section = None
        for _, ip, _, module in frames:
            try:
                same_module = Path(module).resolve() == extension_resolved
            except OSError:
                same_module = False
            if not same_module:
                continue
            line = source_line(
                extension, ip, base, source_path.name, line_cache
            )
            if line is not None:
                section = section_for_line(line, markers)
                break
        if section is None:
            unclassified += 1
        else:
            samples[section] += 1

    return samples, len(frames_by_sample), unclassified, base


def write_results(output_path, samples, total, render_ms, counts):
    rows = []
    for section in SECTIONS:
        sample_count = samples[section]
        share = sample_count / total if total else 0.0
        execution_count = counts[COUNT_KEYS[section]]
        estimated_ms = render_ms * share
        ns_per_call = (
            estimated_ms * 1_000_000.0 / execution_count
            if execution_count
            else 0.0
        )
        rows.append(
            {
                "section": section,
                "samples": sample_count,
                "share_percent": share * 100.0,
                "estimated_ms": estimated_ms,
                "execution_count": execution_count,
                "estimated_ns_per_call": ns_per_call,
            }
        )
    rows.sort(key=lambda row: row["estimated_ms"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "section",
        "samples",
        "share_percent",
        "estimated_ms",
        "execution_count",
        "estimated_ns_per_call",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsys-report",
        default="results/bicycle_cpu_backward_render_nsys.nsys-rep",
    )
    parser.add_argument(
        "--profile-csv",
        default="results/bicycle_cpu_backward_render_profile.csv",
    )
    parser.add_argument(
        "--extension",
        default="submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-310-aarch64-linux-gnu.so",
    )
    parser.add_argument(
        "--source",
        default="submodules/diff-gaussian-rasterization/cuda_rasterizer/backward_cpu.cpp",
    )
    parser.add_argument(
        "--output",
        default="results/bicycle_cpu_backward_render_sections.csv",
    )
    args = parser.parse_args()

    report_path = Path(args.nsys_report).resolve()
    profile_path = Path(args.profile_csv)
    extension = Path(args.extension).resolve()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output)

    render_ms, counts = read_profile(profile_path)
    markers = read_markers(source_path)
    with tempfile.TemporaryDirectory(prefix="backward-render-nsys-") as temp_dir:
        sqlite_path = Path(temp_dir) / "report.sqlite"
        subprocess.run(
            [
                "nsys",
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                f"--output={sqlite_path}",
                str(report_path),
            ],
            check=True,
        )
        samples, total, unclassified, base = analyze(
            sqlite_path, extension, source_path, markers
        )

    rows = write_results(output_path, samples, total, render_ms, counts)
    print(f"render_cpu median: {render_ms:.3f} ms")
    print(f"samples: {total:,}, unclassified: {unclassified:,}")
    print(f"extension load base: 0x{base:x}")
    for row in rows:
        print(
            f"{row['rank']:2d} {row['section']:30s} "
            f"{row['estimated_ms']:9.3f} ms {row['share_percent']:7.3f}%"
        )
    print(f"section CSV: {output_path}")


if __name__ == "__main__":
    main()
