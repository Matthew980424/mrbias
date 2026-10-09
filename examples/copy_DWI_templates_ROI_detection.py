r"""Create the Philips MRBIAS DWI template and session candidates.

The default invocation uses the M271007A ADC data:

    python copy_DWI_templates_ROI_detection.py --dry-run

For every session folder (s1, s2, ...), the script finds the first numeric
scan, excludes non-image DICOM objects, and copies every seventh image
starting at the first image.  With seven b-values, this selects the 15 b=0
images.  The s1 candidate is also copied into the template ``dicom`` folder.
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
    r"E:\project\4dphantom\MRBias_250211\data\M271007A\ADC"
)
DEFAULT_TEMPLATE_ROOT = Path(
    r"E:\project\4dphantom\MRBias_250211\mrbias\roi_detection_templates"
)
TEMPLATE_NAME = "philips_mrlinac_1p5T_emer_m261007A_dwtemp"
SCAN_B_VALUES = 7
SLICE_COUNT = 15
SLICE_INDEX = 8
ROI_RADIUS_MM = 8.0
ROI_HEIGHT_MM = 10.0

FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<measurement>\d{19})_"
    r"(?P<series>\d+)_"
    r"(?P<instance>\d+)_"
    r"(?P<suffix>.+)\.dcm$",
    re.IGNORECASE,
)

# Approximate normalized positions from the existing DWI template.  They are
# used only to constrain the image-based search to the central two tubes.
REFERENCE_CENTRES = ((97.0 / 224.0, 112.0 / 224.0), (122.0 / 224.0, 112.0 / 224.0))
DETECTION_RADIUS_PIXELS = 10.0
DETECTION_SEARCH_RADIUS_PIXELS = 10
DETECTION_ROW_TOLERANCE_PIXELS = 4
DETECTION_SPACING = 25
DETECTION_PRIOR_PENALTY = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the M261007A DWI MRBIAS ROI template."
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help=f"ADC folder containing s1, s2, ... (default: {DEFAULT_DATA_FOLDER}).",
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
        help="Report planned copies and detections without writing files.",
    )
    return parser.parse_args()


def parsed_filename(path: Path) -> re.Match[str]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise RuntimeError(f"Unexpected DICOM filename format: {path.name}")
    return match


def session_folders(data_folder: Path) -> list[Path]:
    folders = sorted(
        (
            path
            for path in data_folder.iterdir()
            if path.is_dir() and re.fullmatch(r"s\d+", path.name, re.IGNORECASE)
        ),
        key=lambda path: (len(path.name), path.name.lower()),
    )
    if not folders:
        raise RuntimeError(f"No session folders named s1, s2, ... found in {data_folder}")
    return folders


def grouped_scans(session_folder: Path) -> dict[int, list[Path]]:
    scans: dict[int, list[Path]] = {}
    for path in session_folder.glob("*.dcm"):
        match = FILENAME_RE.match(path.name)
        if match is None:
            continue
        scans.setdefault(int(match.group("series")), []).append(path)

    for files in scans.values():
        files.sort(key=lambda path: int(parsed_filename(path).group("instance")))
    return scans


def image_files(files: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in files:
        dataset = pydicom.dcmread(path, stop_before_pixels=False)
        if "PixelData" in dataset:
            images.append(path)
    return images


def first_scan_b0_files(session_folder: Path) -> tuple[int, list[Path]]:
    scans = grouped_scans(session_folder)
    if not scans:
        raise RuntimeError(f"No matching DICOM files found in {session_folder}")

    scan_number = min(scans)
    images = image_files(scans[scan_number])
    selected = images[0::SCAN_B_VALUES]
    if len(selected) != SLICE_COUNT:
        raise RuntimeError(
            f"{session_folder.name} scan {scan_number}: expected "
            f"{SLICE_COUNT} b=0 images, found {len(selected)}."
        )
    return scan_number, selected


def load_slice(b0_files: list[Path], slice_index: int) -> np.ndarray:
    if slice_index >= len(b0_files):
        raise RuntimeError(
            f"Requested slice {slice_index}, but only {len(b0_files)} b=0 images exist."
        )
    dataset = pydicom.dcmread(b0_files[slice_index])
    return dataset.pixel_array.astype(np.float32)


def circular_gradient_score(
    gradient: np.ndarray, x: float, y: float, radius: float
) -> float:
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    samples = ndimage.map_coordinates(
        gradient,
        [
            y + radius * np.sin(angles),
            x + radius * np.cos(angles),
        ],
        order=1,
        mode="nearest",
    )
    return float(samples.mean())


def detect_central_centres(image: np.ndarray) -> list[tuple[int, int]]:
    height, width = image.shape
    smoothed = ndimage.gaussian_filter(image, sigma=1.0)
    gradient = np.hypot(
        ndimage.sobel(smoothed, axis=1),
        ndimage.sobel(smoothed, axis=0),
    )
    candidate_sets: list[list[tuple[float, int, int]]] = []
    for reference_x, reference_y in REFERENCE_CENTRES:
        expected_x = reference_x * width
        expected_y = reference_y * height
        candidate_sets.append(
            [
                (
                    circular_gradient_score(
                        gradient, x, y, DETECTION_RADIUS_PIXELS
                    ),
                    x,
                    y,
                )
                for x in range(
                    round(expected_x) - DETECTION_SEARCH_RADIUS_PIXELS,
                    round(expected_x) + DETECTION_SEARCH_RADIUS_PIXELS + 1,
                )
                for y in range(
                    round(expected_y) - DETECTION_SEARCH_RADIUS_PIXELS,
                    round(expected_y) + DETECTION_SEARCH_RADIUS_PIXELS + 1,
                )
            ]
        )

    best_pair: tuple[float, tuple[int, int], tuple[int, int]] | None = None
    for first, second in (
        (first, second)
        for first in candidate_sets[0]
        for second in candidate_sets[1]
    ):
        first_x, first_y = first[1:]
        second_x, second_y = second[1:]
        spacing = second_x - first_x
        if abs(first_y - second_y) > DETECTION_ROW_TOLERANCE_PIXELS:
            continue
        if not 18 <= spacing <= 32:
            continue

        prior_distance = (
            (first_x - REFERENCE_CENTRES[0][0] * width) ** 2
            + (first_y - REFERENCE_CENTRES[0][1] * height) ** 2
            + (second_x - REFERENCE_CENTRES[1][0] * width) ** 2
            + (second_y - REFERENCE_CENTRES[1][1] * height) ** 2
        )
        score = (
            first[0]
            + second[0]
            - DETECTION_PRIOR_PENALTY * prior_distance
            - abs(spacing - DETECTION_SPACING) * DETECTION_PRIOR_PENALTY
        )
        pair = (score, (first_x, first_y), (second_x, second_y))
        if best_pair is None or pair[0] > best_pair[0]:
            best_pair = pair

    if best_pair is None:
        raise RuntimeError("The two central DWI tube centres could not be separated.")
    return [best_pair[1], best_pair[2]]


def roi_yaml(session_centres: dict[str, list[tuple[int, int]]]) -> str:
    lines: list[str] = []
    for session_index, (session, centres) in enumerate(session_centres.items()):
        comment = "" if session_index == 0 else "# "
        lines.append(f"{comment}# {session}")
        for roi_index, (x, y) in enumerate(centres, start=1):
            lines.extend(
                [
                    f"{comment}dw_roi_{roi_index}:",
                    f'{comment}  roi_type: "cylinder"',
                    f"{comment}  roi_radius_mm: {ROI_RADIUS_MM:.1f}",
                    f"{comment}  roi_height_mm: {ROI_HEIGHT_MM:.1f}",
                    f"{comment}  ctr_vox_coords: [{x}, {y}, {SLICE_INDEX}]",
                    "",
                ]
            )
    return "\n".join(lines)


def copy_files(
    files: list[Path],
    destination: Path,
    dry_run: bool,
    clear_existing: bool = False,
) -> None:
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        if clear_existing:
            for existing in destination.glob("*.dcm"):
                existing.unlink()
    for source in files:
        target = destination / source.name
        if dry_run:
            print(f"  {source.name} -> {target}")
        else:
            shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"ADC data folder does not exist: {args.data_folder}")

    sessions = session_folders(args.data_folder)
    if len(sessions) < 2:
        raise SystemExit(
            f"Expected multiple scan sessions, but found only {len(sessions)} in {args.data_folder}"
        )

    template_folder = args.template_root / TEMPLATE_NAME
    candidate_root = template_folder / "candidate template"
    session_centres: dict[str, list[tuple[int, int]]] = {}
    session_b0_files: dict[str, list[Path]] = {}

    for session_folder in sessions:
        scan_number, b0_files = first_scan_b0_files(session_folder)
        centres = detect_central_centres(load_slice(b0_files, SLICE_INDEX))
        session_centres[session_folder.name] = centres
        session_b0_files[session_folder.name] = b0_files
        print(
            f"{session_folder.name}: scan {scan_number}, "
            f"copying {len(b0_files)} b=0 files, detected {centres}"
        )
        copy_files(
            b0_files,
            candidate_root / session_folder.name,
            dry_run=args.dry_run,
            clear_existing=True,
        )

    default_dicom = template_folder / "dicom"
    default_files = session_b0_files[sessions[0].name]
    print(
        f"Default dicom: copying {len(default_files)} b=0 files "
        f"from candidate template/{sessions[0].name}"
    )
    copy_files(
        default_files,
        default_dicom,
        dry_run=args.dry_run,
        clear_existing=True,
    )

    yaml_path = template_folder / "default_DW_rois.yaml"
    if args.dry_run:
        print(f"Would create {yaml_path}")
        print(roi_yaml(session_centres))
    else:
        template_folder.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(roi_yaml(session_centres), encoding="utf-8")
        print(f"Created {yaml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
