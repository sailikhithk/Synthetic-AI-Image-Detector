#!/usr/bin/env python3
"""Download eval data from HuggingFace and extract images for SAI evaluation.

Downloads:
- 200 real images from MS-COCO (bitmind/MS-COCO)
- 200 BigGAN fakes from GenImage (bitmind/GenImage_BigGAN)
- 200 ADM fakes from GenImage (bitmind/GenImage_ADM)

Usage:
    python3 scripts/download_eval_data.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"
N = 200
SIZE = 256


def extract_from_parquet(parquet_path: Path, out_dir: Path, prefix: str, n: int = N,
                         image_field: str = "image", dedupe_field: str | None = None):
    """Extract up to n images from a parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(parquet_path))
    seen = set()
    count = 0
    for rg in range(pf.metadata.num_row_groups):
        if count >= n:
            break
        table = pf.read_row_group(rg)
        for i in range(table.num_rows):
            if count >= n:
                break
            row = {k: table.column(k)[i].as_py() for k in table.column_names}
            if dedupe_field and dedupe_field in row:
                key = row[dedupe_field]
                if key in seen:
                    continue
                seen.add(key)
            img_val = row.get(image_field)
            if isinstance(img_val, dict):
                img_bytes = img_val.get("bytes", b"")
            elif isinstance(img_val, bytes):
                img_bytes = img_val
            else:
                continue
            if len(img_bytes) < 100:
                continue
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
            img.save(out_dir / f"{prefix}_{count:04d}.png")
            count += 1
    print(f"  Extracted {count} images to {out_dir}")
    return count


def main():
    print("Downloading eval data from HuggingFace...")

    # BigGAN
    print("\n[1/3] Downloading GenImage BigGAN shard...")
    path = hf_hub_download(
        repo_id="bitmind/GenImage_BigGAN",
        filename="data/train-00000-of-00008.parquet",
        repo_type="dataset",
        local_dir=str(DATA.parent / "biggan"),
    )
    extract_from_parquet(Path(path), DATA / "biggan", "biggan")

    # ADM
    print("\n[2/3] Downloading GenImage ADM shard...")
    path = hf_hub_download(
        repo_id="bitmind/GenImage_ADM",
        filename="data/train-00000-of-00046.parquet",
        repo_type="dataset",
        local_dir=str(DATA.parent / "adm"),
    )
    extract_from_parquet(Path(path), DATA / "adm", "adm")

    # MS-COCO (real images)
    print("\n[3/3] Downloading MS-COCO real images shard...")
    path = hf_hub_download(
        repo_id="bitmind/MS-COCO",
        filename="data/test-00000-of-00009.parquet",
        repo_type="dataset",
        local_dir=str(DATA.parent / "coco"),
    )
    extract_from_parquet(Path(path), DATA / "real", "real", dedupe_field="cocoid")

    print(f"\nDone. Eval data in {DATA}/")
    print(f"  real/   - {len(list((DATA / 'real').glob('*.png')))} real images")
    print(f"  biggan/ - {len(list((DATA / 'biggan').glob('*.png')))} BigGAN fakes")
    print(f"  adm/    - {len(list((DATA / 'adm').glob('*.png')))} ADM fakes")


if __name__ == "__main__":
    main()
