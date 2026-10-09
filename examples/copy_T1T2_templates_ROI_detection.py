r"""Create the Philips MRBIAS T1/T2 ROI-detection template.

The default invocation uses the M271007A T1/T2 data prepared by
``copy_mrbias_dicom_scans.py``:

    python copy_templates_ROI_detection.py --dry-run

The script copies scan 201 into the template ``dicom`` folder and detects the
11 circular tube centres in slice 38.  The detected centres are written to
both ``default_T1_rois.yaml`` and ``default_T2_rois.yaml``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage


DEFAULT_DATA_FOLDER = Path(
    r"E:\project\4dphantom\MRBias_250211\data\M271007A\T1T2"
)
DEFAULT_TEMPLATE_ROOT = Path(
    r"E:\project\4dphantom\MRBias_250211\mrbias\roi_detection_templates"
)
TEMPLATE_NAME = "philips_mrlinac_1p5T_emer_m261007A"
DW_TEMPLATE_NAME = "philips_mrlinac_1p5T_emer_m261007A_dwtemp"
SCAN_NUMBER = 201
SLICE_INDEX = 38
ROI_RADIUS_MM = 8.0
ROI_HEIGHT_MM = 100.0

FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<measurement>\d{19})_"
    r"(?P<series>\d+)_"
    r"(?P<instance>\d+)_"
    r"(?P<suffix>.+)\.dcm$",
    re.IGNORECASE,
)

# This is the physical tube arrangement, expressed as (row, column) slots.
# Rows contain 3, 4, and 4 tubes respectively.  The order matches the
# existing m260723A ROI YAML files.
ROI_LAYOUT = (
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (1, 1),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 3),
    (1, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the M261007A T1/T2 MRBIAS ROI template."
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help=f"Folder containing T1/T2 DICOM files (default: {DEFAULT_DATA_FOLDER}).",
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        default=DEFAULT_TEMPLATE_ROOT,
        help=f"ROI template root (default: {DEFAULT_TEMPLATE_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and report the planned output without writing files.",
    )
    return parser.parse_args()


def series_number(path: Path) -> int | None:
    match = FILENAME_RE.match(path.name)
    return int(match.group("series")) if match else None


def scan_files(data_folder: Path, scan_number: int) -> list[Path]:
    files = sorted(
        (
            path
            for path in data_folder.glob("*.dcm")
            if series_number(path) == scan_number
        ),
        key=lambda path: path.name,
    )
    if not files:
        raise RuntimeError(f"No DICOM files found for scan {scan_number} in {data_folder}")
    return files


def image_files(files: list[Path]) -> list[tuple[int, Path]]:
    images: list[tuple[int, Path]] = []
    for path in files:
        dataset = pydicom.dcmread(path, stop_before_pixels=False)
        if "PixelData" in dataset:
            images.append((int(dataset.InstanceNumber), path))
    images.sort()
    return images


def load_slice(files: list[Path], slice_index: int) -> tuple[np.ndarray, tuple[float, float]]:
    images = image_files(files)
    if slice_index >= len(images):
        raise RuntimeError(
            f"Requested slice {slice_index}, but scan contains only {len(images)} image slices."
        )

    _, path = images[slice_index]
    dataset = pydicom.dcmread(path)
    pixel_spacing = tuple(float(value) for value in dataset.PixelSpacing)
    return dataset.pixel_array.astype(np.float32), pixel_spacing


def circle_score_map(image: np.ndarray) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(image, sigma=1.0)
    gradient = np.hypot(
        ndimage.sobel(smoothed, axis=1),
        ndimage.sobel(smoothed, axis=0),
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    candidates: list[tuple[float, int, int]] = []

    # The tube grid is in the central phantom area.  Limiting the search
    # avoids selecting the container boundary, which also has strong edges.
    for y in range(250, min(image.shape[0] - 35, 500), 3):
        for x in range(250, min(image.shape[1] - 35, 550), 3):
            scores = [
                ndimage.map_coordinates(
                    gradient,
                    [
                        y + radius * np.sin(angles),
                        x + radius * np.cos(angles),
                    ],
                    order=1,
                    mode="nearest",
                ).mean()
                for radius in (22, 24, 26)
            ]
            candidates.append((max(scores), x, y))

    candidates.sort(reverse=True)
    selected: list[tuple[float, int, int]] = []
    for candidate in candidates:
        _, x, y = candidate
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= 40**2 for _, old_x, old_y in selected):
            selected.append(candidate)
        if len(selected) == len(ROI_LAYOUT):
            break

    if len(selected) != len(ROI_LAYOUT):
        raise RuntimeError(f"Could detect only {len(selected)} of 11 tube centres.")
    return np.array([(x, y) for _, x, y in selected], dtype=float)


def arrange_centres(candidates: np.ndarray) -> list[tuple[int, int]]:
    y_order = np.argsort(candidates[:, 1])
    sorted_candidates = candidates[y_order]
    rows = [
        sorted(sorted_candidates[:3], key=lambda point: point[0]),
        sorted(sorted_candidates[3:7], key=lambda point: point[0]),
        sorted(sorted_candidates[7:], key=lambda point: point[0]),
    ]

    by_slot = {(row, column): point for row, points in enumerate(rows) for column, point in enumerate(points)}
    if any(slot not in by_slot for slot in ROI_LAYOUT):
        raise RuntimeError("Detected tube centres do not form the expected 3/4/4 layout.")
    return [(int(round(by_slot[slot][0])), int(round(by_slot[slot][1]))) for slot in ROI_LAYOUT]


def roi_yaml(prefix: str, centres: list[tuple[int, int]]) -> str:
    lines: list[str] = []
    for index, (x, y) in enumerate(centres, start=1):
        lines.extend(
            [
                f"{prefix}_roi_{index}:",
                ' roi_type: "cylinder"',
                f" roi_radius_mm: {ROI_RADIUS_MM:.1f}",
                f" roi_height_mm: {ROI_HEIGHT_MM:.1f}",
                f" ctr_vox_coords: [{x}, {y}, {SLICE_INDEX}]",
                "",
            ]
        )
    return "\n".join(lines)


def copy_template_dicom(files: list[Path], destination: Path, dry_run: bool) -> None:
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = destination / source.name
        if dry_run:
            print(f"  {source.name} -> {target}")
        else:
            shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"Data folder does not exist: {args.data_folder}")

    files = scan_files(args.data_folder, SCAN_NUMBER)
    image, _ = load_slice(files, SLICE_INDEX)
    centres = arrange_centres(circle_score_map(image))
    print(f"Detected centres on slice {SLICE_INDEX}: {centres}")

    template_folder = args.template_root / TEMPLATE_NAME
    dw_template_folder = args.template_root / DW_TEMPLATE_NAME
    dicom_folder = template_folder / "dicom"
    print(f"Copying {len(files)} scan-{SCAN_NUMBER} DICOM files to {dicom_folder}")

    if not args.dry_run:
        template_folder.mkdir(parents=True, exist_ok=True)
        dw_template_folder.mkdir(parents=True, exist_ok=True)
        copy_template_dicom(files, dicom_folder, dry_run=False)
        (template_folder / "default_T1_rois.yaml").write_text(
            roi_yaml("t1", centres),
            encoding="utf-8",
        )
        (template_folder / "default_T2_rois.yaml").write_text(
            roi_yaml("t2", centres),
            encoding="utf-8",
        )
        print(f"Created {template_folder}")
    else:
        print("Would create:")
        print(f"  {template_folder / 'default_T1_rois.yaml'}")
        print(f"  {template_folder / 'default_T2_rois.yaml'}")
        print(f"  {dw_template_folder}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
