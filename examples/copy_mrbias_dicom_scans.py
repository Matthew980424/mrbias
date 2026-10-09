r"""Copy selected DICOM series into the MRBIAS data layout.

The default paths match the M261007A source and M271007A target described in
the example below:

    python copy_mrbias_dicom_scans.py --dry-run

The source folder is expected to contain DICOM files whose names include the
series number in this form:

    ..._<measurement>_<series>_<instance>_<suffix>.dcm

For example, ``..._2301_000001_...dcm`` belongs to series 2301.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pydicom


DEFAULT_SOURCE = Path(r"D:\DICOM\dicomserver150e\data\M261007A")
DEFAULT_TARGET = Path(r"E:\project\4dphantom\MRBias_250211\data\M271007A")

SERIES_RE = re.compile(r"_(?P<series>\d+)_\d+_[^/\\]+\.dcm$", re.IGNORECASE)

T1T2_SERIES = tuple(range(201, 2000, 100))
ADC_SERIES_BY_SESSION: dict[str, tuple[int, ...]] = {
    "s1": (2001, 2101),
    "s2": (2301, 2401),
    "s3": (2601, 2701),
    "s4": (2901, 3001),
    "s5": (3201, 3301),
    "s6": (3501, 3601),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the requested DICOM scan series into the MRBIAS layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Flat source DICOM folder (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Destination folder (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned copies without creating folders or copying files.",
    )
    return parser.parse_args()


def series_number(path: Path) -> int | None:
    match = SERIES_RE.search(path.name)
    return int(match.group("series")) if match else None


def is_image_dicom(path: Path) -> bool:
    dataset = pydicom.dcmread(path, stop_before_pixels=False)
    return "PixelData" in dataset


def group_dicom_files(source: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in source.iterdir():
        if not path.is_file() or path.suffix.lower() != ".dcm":
            continue
        series = series_number(path)
        if series is not None and is_image_dicom(path):
            grouped[series].append(path)

    for files in grouped.values():
        files.sort(key=lambda path: path.name)
    return grouped


def copy_series(
    grouped_files: dict[int, list[Path]],
    source: Path,
    destination: Path,
    series: tuple[int, ...],
    dry_run: bool,
) -> int:
    missing = [number for number in series if number not in grouped_files]
    if missing:
        missing_text = ", ".join(str(number) for number in missing)
        raise RuntimeError(f"Series not found in {source}: {missing_text}")

    files = [path for number in series for path in grouped_files[number]]
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    for path in files:
        target = destination / path.name
        if dry_run:
            print(f"  {path} -> {target}")
        else:
            shutil.copy2(path, target)
    return len(files)


def main() -> int:
    args = parse_args()
    source = args.source
    target = args.target

    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    grouped_files = group_dicom_files(source)
    if not grouped_files:
        raise SystemExit(f"No matching DICOM files found in: {source}")

    jobs = [(target / "T1T2", T1T2_SERIES)]
    jobs.extend(
        (target / "ADC" / session, series)
        for session, series in ADC_SERIES_BY_SESSION.items()
    )

    total = 0
    for destination, series in jobs:
        print(f"{destination.name if destination.name != 'T1T2' else 'T1T2'}:")
        copied = copy_series(
            grouped_files,
            source,
            destination,
            series,
            dry_run=args.dry_run,
        )
        print(f"  {'Would copy' if args.dry_run else 'Copied'} {copied} file(s).")
        total += copied

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {total} DICOM file(s) into {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
