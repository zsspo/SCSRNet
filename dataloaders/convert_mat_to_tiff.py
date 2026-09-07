import argparse
import os
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile as tiff


def natural_key(name: str):
    match = re.search(r"(.*?)(\d+)$", name)
    if not match:
        return (name, -1)
    prefix, num = match.group(1), int(match.group(2))
    return (prefix, num)


def collect_samples(src_root: Path):
    samples = []
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        mat_files = list(entry.glob("*.mat"))
        if not mat_files:
            continue
        for mat_path in mat_files:
            if mat_path.name.lower().endswith("_gt.mat"):
                continue
            sample_name = mat_path.stem
            samples.append((sample_name, mat_path))
    samples.sort(key=lambda x: natural_key(x[0]))
    return samples


def ensure_dirs(out_root: Path):
    for sub in ("hsi", "ref", "rgb"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def load_mat_arrays(mat_path: Path):
    mat = sio.loadmat(mat_path)
    if "y" not in mat or "ref" not in mat or "msi" not in mat:
        raise KeyError(f"Missing keys in {mat_path.name}. Expected: y, ref, msi")
    return mat["y"], mat["ref"], mat["msi"]


def rgb_to_uint8(rgb: np.ndarray, max_value: float):
    if max_value <= 0:
        max_value = float(np.max(rgb)) if rgb.size else 1.0
    scaled = np.clip(rgb / max_value * 255.0, 0.0, 255.0)
    return np.rint(scaled).astype(np.uint8)


def write_tiff(path: Path, array: np.ndarray, rgb=False):
    if rgb:
        tiff.imwrite(path, array, photometric="rgb", planarconfig="contig", metadata=None)
    else:
        tiff.imwrite(path, array, planarconfig="contig", metadata=None)


def convert_dataset(
    name: str,
    src_root: Path,
    out_root: Path,
    rgb_mode: str,
    max_value: float,
    overwrite: bool,
    dry_run: bool,
):
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    samples = collect_samples(src_root)
    if not samples:
        raise RuntimeError(f"No .mat samples found in {src_root}")

    ensure_dirs(out_root)

    name_to_index = {sample_name: idx for idx, (sample_name, _) in enumerate(samples)}

    for idx, (sample_name, mat_path) in enumerate(samples):
        out_hsi = out_root / "hsi" / f"{idx}.tif"
        out_ref = out_root / "ref" / f"{idx}.tif"
        out_rgb = out_root / "rgb" / f"{idx}.tif"

        if not overwrite and out_hsi.exists() and out_ref.exists() and out_rgb.exists():
            continue

        if dry_run:
            continue

        y, ref, msi = load_mat_arrays(mat_path)

        hsi = np.asarray(y, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)

        if rgb_mode == "float32":
            rgb = np.asarray(msi, dtype=np.float32)
        else:
            rgb = rgb_to_uint8(np.asarray(msi, dtype=np.float32), max_value)

        write_tiff(out_hsi, hsi, rgb=False)
        write_tiff(out_ref, ref, rgb=False)
        write_tiff(out_rgb, rgb, rgb=True)

    for split in ("train", "val", "test"):
        split_file = src_root / f"{split}.txt"
        if not split_file.exists():
            continue
        names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        indices = [str(name_to_index[n]) for n in names if n in name_to_index]
        if not indices:
            continue
        out_split = out_root / f"{split}.txt"
        if not dry_run:
            out_split.write_text("\n".join(indices) + "\n", encoding="utf-8")

    print(f"[{name}] samples: {len(samples)}")
    print(f"[{name}] output: {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chikusei/Pavia .mat HSI datasets to TIFF with Daxing-like folder structure."
    )
    parser.add_argument("--dataset", choices=["chikusei", "pavia", "all"], default="pavia")
    parser.add_argument("--src-root", type=str, default=None, help="Override source root path.")
    parser.add_argument("--out-root", type=str, default="H:\\code\\HyperLKN\\fusion_rgb\\datasets_new\\Pavia", help="Override output root path.")
    parser.add_argument("--rgb-mode", choices=["uint8", "float32"], default="float32")
    parser.add_argument("--max-value", type=float, default=None, help="Max value for RGB scaling.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    datasets_root = project_root / "datasets"

    def defaults_for(ds_name: str):
        if ds_name == "chikusei":
            return datasets_root / "Chikusei", datasets_root / "Chikusei_tiff", 15133.0
        if ds_name == "pavia":
            return datasets_root / "Pavia", datasets_root / "Pavia_tiff", 8000.0
        raise ValueError(ds_name)

    targets = ["chikusei", "pavia"] if args.dataset == "all" else [args.dataset]
    for ds_name in targets:
        src_default, out_default, max_default = defaults_for(ds_name)
        src_root = Path(args.src_root) if args.src_root else src_default
        out_root = Path(args.out_root) if args.out_root and len(targets) == 1 else out_default
        max_value = args.max_value if args.max_value is not None else max_default

        convert_dataset(
            ds_name,
            src_root,
            out_root,
            args.rgb_mode,
            max_value,
            args.overwrite,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
