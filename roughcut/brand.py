"""BrandKit load/save + interactive setup."""

from __future__ import annotations

import json
from pathlib import Path

from roughcut.schemas import BrandKit

DEFAULT_BRAND_PATH = Path("brand.json")


def load_brand(path: Path) -> BrandKit:
    """Load a BrandKit JSON file."""
    return BrandKit.model_validate_json(path.read_text())


def save_brand(brand: BrandKit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brand.model_dump_json(indent=2))


def init_default(path: Path = DEFAULT_BRAND_PATH) -> BrandKit:
    """Write a starter BrandKit with no logo. User fills it in."""
    b = BrandKit(name="my-brand")
    save_brand(b, path)
    return b
