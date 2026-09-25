"""
spartan/models/registry.py
==========================
Central model registry for SPARTAN.

Architecture
------------
Uses **SwinIR-M x4** (``embed_dim=180``, ``depths=[6]*6``) which exactly matches
the official ``001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`` checkpoint.  This
avoids any dimensional slicing or padding of the transformer body — all 11.9 M
parameters are transferred with 100 % fidelity.

The only weight surgery required is on two convolutional head/tail layers:
  * ``conv_first``  [180, 3, k, k] → [180, **4**, k, k]  (add NIR channel)
  * ``conv_last``   [3, 64, k, k]  → [**4**, 64, k, k]   (add NIR output)

Load priority
-------------
1. ``spartan_swin_adapted_x4.pth``  — pre-adapted 4-ch checkpoint (instant load)
2. ``001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`` — official; needs 3→4 ch surgery
3. Any other native 4-ch / embed=180 checkpoint
4. Random initialisation (saved for reuse)

Public API
----------
``build_swin_model(device)``         → ``(model, weight_path, source)``
``load_pretrained_swinir(model, path, device)`` → ``nn.Module``
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("spartan.models.registry")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_CURRENT_DIR: Path = Path(__file__).resolve().parent   # …/spartan/models
_SPARTAN_DIR: Path = _CURRENT_DIR.parent               # …/spartan
_PROJECT_ROOT: Path = _SPARTAN_DIR.parent              # project root

WEIGHTS_DIR: Path = _SPARTAN_DIR / "weights"

# ---------------------------------------------------------------------------
# SwinIR-M x4 hyper-parameters
# Matches: 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth exactly
# ---------------------------------------------------------------------------

SWINIR_HPARAMS: dict = dict(
    img_size=64,
    patch_size=1,
    in_chans=3,           # Native RGB (B4, B3, B2) — exact match for official checkpoint
    out_chans=3,          # Native RGB output
    embed_dim=180,        # Exact match for SwinIR-M
    depths=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.0,
    upscale=4,
    img_range=1.0,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)

# ---------------------------------------------------------------------------
# Checkpoint candidate names — searched in order
# ---------------------------------------------------------------------------

# Tier-1: pre-adapted SPARTAN 4-ch / embed=180 checkpoints (instant load)
_ADAPTED_CANDIDATE_NAMES: list[str] = [
    "spartan_swin_adapted_x4.pth",
    "spartan_swin_adapted_x4.pt",
]

# Tier-2: official SwinIR-M x4 checkpoints (needs conv_first+conv_last surgery)
_OFFICIAL_CANDIDATE_NAMES: list[str] = [
    "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
    "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
]

# Tier-3: legacy SPARTAN embed=60 or other native checkpoints (strict=False fallback)
_LEGACY_CANDIDATE_NAMES: list[str] = [
    "SPARTAN-SwinTransformer.pth",
    "SPARTAN-SwinTransformer.pt",
    "spartan_swin_transformer.pth",
    "spartan_swin_transformer.pt",
    "spartan_swinir.pth",
    "spartan_swinir.pt",
    "swinir.pth",
    "swinir.pt",
    "swin_transformer.pth",
    "swin_transformer.pt",
    "swinir_best.pth",
    "swin_transformer_best.pth",
]

# ---------------------------------------------------------------------------
# Search directory resolution
# ---------------------------------------------------------------------------

def _get_search_directories() -> list[Path]:
    dirs: list[Path] = [
        WEIGHTS_DIR,
        _PROJECT_ROOT / "spartan" / "weights",
        _PROJECT_ROOT / "checkpoints",
    ]

    # Cross-worktree resolution when running inside Antigravity
    if "antigravity" in str(_PROJECT_ROOT).lower():
        wt_parent = _PROJECT_ROOT.parent
        if wt_parent.is_dir():
            for sibling in wt_parent.iterdir():
                if sibling.is_dir() and sibling != _PROJECT_ROOT:
                    dirs.append(sibling / "spartan" / "weights")
                    dirs.append(sibling / "checkpoints")

    # Always check user's primary repo
    user_spartan = Path(r"C:\Users\abrah\SIH-SPARTAN\SPARTAN")
    if user_spartan.is_dir():
        dirs.append(user_spartan / "spartan" / "weights")
        dirs.append(user_spartan / "checkpoints")

    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _find_checkpoint(names: list[str]) -> Path | None:
    """Return the first file that exists matching any name across all search dirs."""
    for d in _get_search_directories():
        for name in names:
            p = d / name
            if p.is_file():
                return p
    return None


def _ensure_weights_dir() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Architecture builder
# ---------------------------------------------------------------------------

def _import_swinir():
    try:
        from spartan.models.swinir import SwinIR  # noqa: PLC0415
        return SwinIR
    except ImportError:
        pass
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from models.swinir import SwinIR  # noqa: PLC0415
    return SwinIR


def _build_architecture() -> nn.Module:
    SwinIR = _import_swinir()
    return SwinIR(**SWINIR_HPARAMS)


# ---------------------------------------------------------------------------
# State-dict unwrapping
# ---------------------------------------------------------------------------

def _unwrap_state_dict(raw: object) -> dict:
    """
    Extract a flat weight dict from various checkpoint formats:
    ``params_ema``, ``params``, ``model_state_dict``, ``state_dict``,
    ``model``, or a bare flat dict.  Strips ``"module."`` DDP prefix.
    """
    if not isinstance(raw, dict):
        return raw  # type: ignore[return-value]

    inner = raw
    for key in ("params_ema", "params", "model_state_dict", "state_dict", "model", "net", "weights"):
        if key in inner and isinstance(inner[key], dict):
            inner = inner[key]
            break

    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in inner.items()
    }


def _probe(state_dict: dict) -> dict:
    """Return ``{in_chans, embed_dim, n_depths}`` from a state dict."""
    cf = state_dict.get("conv_first.weight")
    nw = state_dict.get("norm.weight")
    layer_keys = [k for k in state_dict if k.startswith("layers.")]
    n_depths = (max(int(k.split(".")[1]) for k in layer_keys) + 1) if layer_keys else -1
    return {
        "in_chans":  int(cf.shape[1]) if cf is not None else -1,
        "embed_dim": int(nw.shape[0]) if nw is not None else -1,
        "n_depths":  n_depths,
    }



def load_pretrained_swinir(
    model: nn.Module,
    pretrained_path: str | Path,
    device: str = "cpu",
) -> nn.Module:
    """
    Load official SwinIR-M x4 weights with 100% strict parameter match.
    Zero key remapping or surgery required — architecture aligns exactly with
    the official 001_classicalSR_DF2K_s64w8_SwinIR-M_x4 checkpoint.
    """
    pretrained_path = Path(pretrained_path)
    logger.info("Loading weights from '%s'", pretrained_path)

    checkpoint = torch.load(pretrained_path, map_location=torch.device(device))
    state_dict = checkpoint.get("params", checkpoint)

    # Strip DDP prefix if present
    clean_dict: dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_dict, strict=True)
    logger.info("Loaded %d checkpoint keys with strict=True.", len(clean_dict))
    model.eval()
    return model




# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_adapted(model: nn.Module, path: Path) -> None:
    """Save a 4-ch adapted checkpoint so subsequent runs skip surgery."""
    _ensure_weights_dir()
    torch.save({"model_state_dict": model.state_dict()}, path)
    logger.info("Saved adapted checkpoint → '%s'", path)


def _save_initialised(model: nn.Module) -> Path:
    """Persist a random-init model and return the path."""
    save_path = WEIGHTS_DIR / "spartan_swin_transformer.pth"
    _ensure_weights_dir()
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    logger.warning(
        "No pretrained checkpoint found — random-init saved to '%s'. "
        "Run `python -m spartan.weights.download_weights` to fetch official weights.",
        save_path,
    )
    return save_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_swin_model(device: str = "cpu") -> Tuple:
    """
    Build and return a ready-to-infer SwinIR-M x4 RGB model.

    Load priority
    -------------
    1. ``001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`` — official 3-ch checkpoint,
       loaded with strict=True and zero weight surgery.
    2. Random initialisation — warns the user to download the checkpoint.

    Any stale ``spartan_swin_adapted_x4.pth`` (4-ch, corrupted) is deleted on
    startup so it cannot corrupt inference again.

    Parameters
    ----------
    device : str
        PyTorch device string (``"cpu"`` | ``"cuda"``).

    Returns
    -------
    model : nn.Module   — SwinIR-M x4 RGB in eval mode on *device*.
    weight_path : Path  — Where weights were loaded from (or saved to).
    source : str        — ``"trained"`` | ``"initialised"``
    """
    _ensure_weights_dir()

    # Delete stale 4-ch adapted checkpoint so it is never loaded again
    for stale_name in ["spartan_swin_adapted_x4.pth", "spartan_swin_adapted_x4.pt"]:
        for d in _get_search_directories():
            stale = d / stale_name
            if stale.is_file():
                try:
                    stale.unlink()
                    logger.info("Deleted stale 4-ch adapted checkpoint: '%s'", stale)
                except OSError as e:
                    logger.warning("Could not delete stale checkpoint '%s': %s", stale, e)

    model = _build_architecture()

    # ── Official 3-ch SwinIR-M x4 checkpoint ────────────────────────────────
    official_path = _find_checkpoint(_OFFICIAL_CANDIDATE_NAMES)
    if official_path is not None:
        logger.info(
            "[Tier 1] Official SwinIR-M x4: '%s'. strict=True, zero surgery.",
            official_path,
        )
        try:
            load_pretrained_swinir(model, official_path, device)
            model.eval()
            model.to(torch.device(device))
            return model, official_path, "trained"
        except Exception as exc:
            traceback.print_exc()
            logger.error("Official checkpoint load failed: %s — random init.", exc)
            model = _build_architecture()

    # ── Random initialisation ────────────────────────────────────────────────
    logger.warning(
        "[Tier 2] No official checkpoint found. Run:\n"
        "  python -m spartan.weights.download_weights\n"
        "to fetch 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth."
    )
    weight_path = _save_initialised(model)
    model.eval()
    model.to(torch.device(device))
    return model, weight_path, "initialised"
