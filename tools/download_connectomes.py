#!/usr/bin/env python3
"""Download open Drosophila connectome & transcriptomics datasets required to build and verify Fly-NCA.

Datasets covered (publicly hosted on Google Cloud Storage, Zenodo, Fly Cell Atlas, and GitHub):
1. BANC v888 (Brain and Nerve Cord, 188,508 cells, Bates et al. 2025/2026) — GCS `lee-lab_brain-and-nerve-cord-fly-connectome`
2. MaleCNS v1.0 & v0.9 (Male-specific dimorphic circuits, Berg et al. 2025) — GCS `flyem-male-cns` & `lee-lab`
3. BANC-project curated metadata, Frankenbrain DCV tables & Shiu et al. (2024) holdout — GitHub `flyconnectome/BANC-project`
4. Fly Cell Atlas 10x stringent scRNA-seq (Head & Body `.h5ad`, Li et al. 2022 Science) — `cloud.flycellatlas.org`
5. FAFB v783 (FlyWire adult brain, Dorkenwald et al. 2024 Nature) — Zenodo record `10676866` (included in `--tier full`)

Usage:
    # Check status of local files without downloading:
    python tools/download_connectomes.py --check

    # Download the essential files required by `tools/build_pan_connectome_189766.py` and `verify_principles.py --ship`:
    python tools/download_connectomes.py --tier build

    # Download all tier-1 datasets (including FAFB v783, NBLAST matrices, and enriched synapse tables):
    python tools/download_connectomes.py --tier full --data-dir ./data/casa-data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """Single downloadable connectome or transcriptomic file."""

    tier: str          # 'build' | 'full'
    category: str      # 'banc-v888' | 'mcns-v1.0' | 'malecns-v0.9' | 'fca' | 'fafb-v783'
    rel_dir: str       # Relative directory inside `data_dir`
    filename: str      # Target file name
    url: str           # Public HTTPS download URL


ASSETS: tuple[DatasetAsset, ...] = (
    # --- Tier 'build': Essential files for `tools/build_pan_connectome_189766.py` & `verify_principles.py` ---
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_meta.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_metrics.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_metrics.feather",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_edgelist_simple_v3.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_edgelist_simple_v3.feather",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_edgelist_split_v3.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_edgelist_split_v3.feather",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_betweenness_afferent_to_efferent_v3.csv",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_betweenness_afferent_to_efferent_v3.csv",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888",
        "banc_888_cns_network_spectral_clustering_v3.csv",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_cns_network_spectral_clustering_v3.csv",
    ),
    DatasetAsset(
        "build",
        "banc-v888",
        "connectomes/banc-v888/influence",
        "influence_all_to_effector_subclass.parquet",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/influence/influence_all_to_effector_subclass.parquet",
    ),
    DatasetAsset(
        "build",
        "banc-v888-extras",
        "connectomes/banc-v888-extras/annotations/v888",
        "somas_v1.parquet",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/neuron_annotations/v888/somas_v1.parquet",
    ),
    DatasetAsset(
        "build",
        "banc-v888-extras",
        "connectomes/banc-v888-extras/annotations/v888",
        "cell_info.parquet",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/neuron_annotations/v888/cell_info.parquet",
    ),
    DatasetAsset(
        "build",
        "banc-v888-extras",
        "connectomes/banc-v888-extras/annotations/v888",
        "mitochondria_v1_human_readable.csv.gz",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/neuron_annotations/v888/mitochondria_v1_human_readable.csv.gz",
    ),
    DatasetAsset(
        "build",
        "mcns-v1.0",
        "connectomes/mcns-v1.0",
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    ),
    DatasetAsset(
        "build",
        "mcns-v1.0",
        "connectomes/mcns-v1.0",
        "body-neurotransmitters-male-cns-v1.0.feather",
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
    ),
    DatasetAsset(
        "build",
        "mcns-v1.0",
        "connectomes/mcns-v1.0",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    ),
    DatasetAsset(
        "build",
        "malecns-v0.9",
        "connectomes/malecns-v0.9",
        "malecns_09_meta.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/malecns_09/malecns_09_meta.feather",
    ),
    DatasetAsset(
        "build",
        "malecns-v0.9",
        "connectomes/malecns-v0.9",
        "malecns_09_split_edgelist.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/malecns_09/malecns_09_split_edgelist.feather",
    ),
    DatasetAsset(
        "build",
        "fca",
        "fca",
        "fca_head_10x_stringent.h5ad",
        "https://cloud.flycellatlas.org/index.php/s/LAEybPc2HZnpzKs/download",
    ),
    DatasetAsset(
        "build",
        "fca",
        "fca",
        "fca_body_10x_stringent.h5ad",
        "https://cloud.flycellatlas.org/index.php/s/nXnbWzQnLxTXmMT/download",
    ),
    # --- Tier 'full': Additional FAFB v783, NBLAST, and enriched synapse tables ---
    DatasetAsset(
        "full",
        "fafb-v783",
        "connectomes/fafb-v783",
        "flywire_synapses_783.feather",
        "https://zenodo.org/api/records/10676866/files/flywire_synapses_783.feather/content",
    ),
    DatasetAsset(
        "full",
        "fafb-v783",
        "connectomes/fafb-v783",
        "proofread_connections_783.feather",
        "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
    ),
    DatasetAsset(
        "full",
        "fafb-v783",
        "connectomes/fafb-v783",
        "per_neuron_neuropil_count_post_783.feather",
        "https://zenodo.org/api/records/10676866/files/per_neuron_neuropil_count_post_783.feather/content",
    ),
    DatasetAsset(
        "full",
        "fafb-v783",
        "connectomes/fafb-v783",
        "per_neuron_neuropil_count_pre_783.feather",
        "https://zenodo.org/api/records/10676866/files/per_neuron_neuropil_count_pre_783.feather/content",
    ),
    DatasetAsset(
        "full",
        "fafb-v783",
        "connectomes/fafb-v783",
        "proofread_root_ids_783.npy",
        "https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content",
    ),
    DatasetAsset(
        "full",
        "banc-v888-extras",
        "connectomes/banc-v888-extras/nblast",
        "banc_mirror_nblast.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/nblast/banc_mirror_nblast.feather",
    ),
    DatasetAsset(
        "full",
        "banc-v888-extras",
        "connectomes/banc-v888-extras/nblast",
        "banc_malecns_v0.9_nblast.feather",
        "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/nblast/banc_malecns_v0.9_nblast.feather",
    ),
)

BANC_PROJECT_GIT_URL = "https://github.com/flyconnectome/BANC-project.git"


def resolve_default_data_dir() -> Path:
    """Resolve data root from FLY_NCA_DATA_DIR or ./data/casa-data."""
    env_dir = os.environ.get("FLY_NCA_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[1] / "data" / "casa-data"


def selected_assets(tier: str) -> list[DatasetAsset]:
    if tier == "full":
        return list(ASSETS)
    return [a for a in ASSETS if a.tier == "build"]


def check_status(data_dir: Path, tier: str) -> int:
    items = selected_assets(tier)
    missing = 0
    print(f"Data directory: {data_dir}")
    print("-" * 92)
    for a in items:
        p = data_dir / a.rel_dir / a.filename
        if p.is_file():
            sz_mb = p.stat().st_size / (1024 * 1024)
            print(f"[OK]      {a.category:16s} | {sz_mb:8.1f} MB | {a.rel_dir}/{a.filename}")
        else:
            missing += 1
            print(f"[MISSING] {a.category:16s} | {'--':>8s}    | {a.rel_dir}/{a.filename}")

    banc_repo = data_dir / "connectomes" / "repos" / "BANC-project"
    if banc_repo.is_dir():
        print(f"[OK]      {'git-repo':16s} | {'cloned':>11s} | connectomes/repos/BANC-project")
    else:
        missing += 1
        print(f"[MISSING] {'git-repo':16s} | {'--':>11s} | connectomes/repos/BANC-project ({BANC_PROJECT_GIT_URL})")
    print("-" * 92)
    print(f"Summary ({tier}): {len(items) + 1 - missing}/{len(items) + 1} present, {missing} missing.")
    return missing


def download_assets(data_dir: Path, tier: str) -> None:
    items = selected_assets(tier)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clone flyconnectome/BANC-project if not present (provides banc_888_meta_20260521.parquet, franken_meta, shiu_et_al_2024)
    banc_repo = data_dir / "connectomes" / "repos" / "BANC-project"
    if not banc_repo.is_dir():
        banc_repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"[git] Cloning {BANC_PROJECT_GIT_URL} -> {banc_repo} ...")
        subprocess.run(["git", "clone", "--depth", "1", BANC_PROJECT_GIT_URL, str(banc_repo)], check=True)

    to_fetch = [a for a in items if not (data_dir / a.rel_dir / a.filename).is_file()]
    if not to_fetch:
        print("All requested files are already present on disk.")
        return

    aria2 = shutil.which("aria2c")
    if aria2:
        manifest_path = data_dir / f"download-{tier}.aria2.txt"
        lines: list[str] = []
        for a in to_fetch:
            target_dir = data_dir / a.rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            lines.append(a.url)
            lines.append(f"  dir={target_dir}")
            lines.append(f"  out={a.filename}")
            lines.append("")
        manifest_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[aria2c] Downloading {len(to_fetch)} files using {manifest_path} ...")
        subprocess.run(
            [
                aria2,
                f"--input-file={manifest_path}",
                "--max-concurrent-downloads=3",
                "--split=8",
                "--min-split-size=10M",
                "--continue=true",
                "--auto-file-renaming=false",
            ],
            check=True,
        )
    else:
        for idx, a in enumerate(to_fetch, start=1):
            target_dir = data_dir / a.rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / a.filename
            tmp_file = target_dir / (a.filename + ".part")
            print(f"[{idx}/{len(to_fetch)}] Downloading {a.url} -> {out_file} ...")
            urllib.request.urlretrieve(a.url, tmp_file)
            tmp_file.replace(out_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download open Drosophila connectome datasets for Fly-NCA.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_default_data_dir(),
        help="Target root directory (default: $FLY_NCA_DATA_DIR or ./data/casa-data)",
    )
    parser.add_argument(
        "--tier",
        choices=("build", "full"),
        default="build",
        help="'build' downloads files needed to compile the 189,766-cell connectome; 'full' adds FAFB v783 & NBLAST",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which files are present/missing on disk without downloading",
    )
    args = parser.parse_args()

    if args.check:
        check_status(args.data_dir, args.tier)
        return 0

    download_assets(args.data_dir, args.tier)
    check_status(args.data_dir, args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
