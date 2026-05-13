from __future__ import annotations

from pathlib import Path

from .utils import DATA_DIR


def delete_temporary_files() -> list[str]:
    deleted: list[str] = []
    for folder in [DATA_DIR / "raw", DATA_DIR / "processed"]:
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.glob("*"):
            if path.name == ".gitkeep" or not path.is_file():
                continue
            path.unlink(missing_ok=True)
            deleted.append(str(path))
    return deleted


def remove_partial_refresh(path: Path) -> None:
    if path.exists():
        path.unlink()
