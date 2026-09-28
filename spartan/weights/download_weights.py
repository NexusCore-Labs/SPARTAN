"""
spartan/weights/download_weights.py
====================================
Utility script to download official SwinIR pretrained x4 super-resolution
checkpoints into ``spartan/weights/`` if they are not already present.

Usage
-----
Run directly::

    python spartan/weights/download_weights.py

Or import and call::

    from spartan.weights.download_weights import download_swinir_pretrained
    path = download_swinir_pretrained()

Downloaded checkpoints
----------------------
``001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth``
    Official SwinIR-M classical SR x4 trained on DF2K (DIV2K + Flickr2K).
    Architecture: embed_dim=180, depths=[6]*6, 3-channel RGB.
    Size: ~64 MB.
    Source: https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0

    When used with SPARTAN (embed=60, 4-ch Sentinel-2), ``registry.py`` will
    automatically apply channel surgery (conv_first 3→4 ch) and embed-dim
    slicing (180→60) via ``load_pretrained_swinir()``.

Note
----
If you have your own trained SPARTAN checkpoint (4-ch, embed=60), place it in
``spartan/weights/`` with any of the native candidate names listed in
``registry.py``.  The registry always prefers native checkpoints over official
ones so your trained weights will be loaded directly without any surgery.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("spartan.weights.downloader")

# ---------------------------------------------------------------------------
# Catalogue of downloadable checkpoints
# ---------------------------------------------------------------------------

WEIGHTS_DIR: Path = Path(__file__).resolve().parent

_CATALOGUE: list[dict] = [
    {
        "name": "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        "url": (
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        ),
        "size_mb": 64,
        "description": "SwinIR-M classical SR x4 (DF2K, 3-ch RGB, embed=180)",
        "sha256": None,  # not verified — update when known
    },
    {
        "name": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        "url": (
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
        ),
        "size_mb": 64,
        "description": "SwinIR-M classical SR x4 (DIV2K, 3-ch RGB, embed=180)",
        "sha256": None,
    },
]

# The recommended default to download
_DEFAULT_CHECKPOINT = _CATALOGUE[0]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(description: str):
    """Return a urlretrieve-compatible reporthook that prints progress."""
    _last_pct = [-1]

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = int(downloaded / total_size * 100)
        else:
            pct = 0
        if pct != _last_pct[0] and pct % 10 == 0:
            mb_done = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024 if total_size > 0 else "?"
            print(
                f"  [{description}]  {mb_done:.0f} / {mb_total:.0f} MB  ({pct}%)",
                flush=True,
            )
            _last_pct[0] = pct

    return hook


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(
    entry: dict,
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a single checkpoint defined by *entry* into *dest_dir*.

    Parameters
    ----------
    entry : dict
        One of the dicts from ``_CATALOGUE``.
    dest_dir : Path, optional
        Directory to save into.  Defaults to ``spartan/weights/``.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Local path of the downloaded file.
    """
    if dest_dir is None:
        dest_dir = WEIGHTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / entry["name"]

    if out_path.is_file() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(
            f"✓  {entry['name']} already present ({size_mb:.0f} MB) — skipping.",
            flush=True,
        )
        return out_path

    print(
        f"Downloading: {entry['description']}\n"
        f"  URL  : {entry['url']}\n"
        f"  Dest : {out_path}\n"
        f"  Size : ~{entry['size_mb']} MB",
        flush=True,
    )

    try:
        tmp_path = out_path.with_suffix(".tmp")
        urllib.request.urlretrieve(
            entry["url"],
            tmp_path,
            reporthook=_progress_hook(entry["name"]),
        )
        tmp_path.rename(out_path)
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Download failed for '{entry['name']}': {exc}\n"
            f"Please download manually from:\n  {entry['url']}\n"
            f"and place it in: {dest_dir}"
        ) from exc

    actual_mb = out_path.stat().st_size / 1024 / 1024
    print(f"✓  Saved {actual_mb:.0f} MB → {out_path}", flush=True)

    if entry.get("sha256"):
        print("  Verifying SHA-256...", flush=True)
        actual = _sha256(out_path)
        if actual != entry["sha256"]:
            out_path.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch for '{entry['name']}': "
                f"expected {entry['sha256']}, got {actual}"
            )
        print("  SHA-256 OK.", flush=True)

    return out_path


def download_swinir_pretrained(
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download the recommended SwinIR-M x4 classical SR checkpoint.

    This is the **default** checkpoint used by ``registry.py`` when no
    native SPARTAN trained weights are found.

    Parameters
    ----------
    dest_dir : Path, optional
        Destination directory.  Defaults to ``spartan/weights/``.
    force : bool
        Re-download even if already present.

    Returns
    -------
    Path
        Local path to the downloaded checkpoint.
    """
    return download_checkpoint(_DEFAULT_CHECKPOINT, dest_dir=dest_dir, force=force)


def download_all(dest_dir: Optional[Path] = None, force: bool = False) -> list[Path]:
    """Download every checkpoint in the catalogue."""
    return [
        download_checkpoint(entry, dest_dir=dest_dir, force=force)
        for entry in _CATALOGUE
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download official SwinIR pretrained x4 checkpoints for SPARTAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default (SwinIR-M DF2K x4):
  python -m spartan.weights.download_weights

  # Download all available checkpoints:
  python -m spartan.weights.download_weights --all

  # Force re-download even if already present:
  python -m spartan.weights.download_weights --force

  # Download to a custom directory:
  python -m spartan.weights.download_weights --dest /path/to/weights
""",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available checkpoints (default: just the recommended one)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists."
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help="Destination directory (default: spartan/weights/)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available checkpoints and exit."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.list:
        print("\nAvailable checkpoints:")
        for i, entry in enumerate(_CATALOGUE):
            marker = " [default]" if i == 0 else ""
            print(f"  {i+1}. {entry['name']}{marker}")
            print(f"       {entry['description']}")
            print(f"       ~{entry['size_mb']} MB")
        sys.exit(0)

    dest_dir = args.dest or WEIGHTS_DIR
    if args.all:
        paths = download_all(dest_dir=dest_dir, force=args.force)
        print(f"\n✓  Downloaded {len(paths)} checkpoint(s).")
    else:
        path = download_swinir_pretrained(dest_dir=dest_dir, force=args.force)
        print(f"\n✓  Checkpoint ready at: {path}")
        print(
            "\nThe registry will now automatically adapt these weights (3-ch → 4-ch,\n"
            "embed_dim 180 → 60) when loading SPARTAN-SwinTransformer.\n"
            "For best results, fine-tune on Sentinel-2 data."
        )


if __name__ == "__main__":
    main()
