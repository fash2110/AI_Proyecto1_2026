from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitPaths:
    train_list: Path | None = None
    val_list: Path | None = None
    test_list: Path | None = None


def _read_list_file(p: Path) -> set[str]:
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return {ln for ln in lines if ln and not ln.startswith("#")}


def resolve_speech_commands_splits(root: str | Path) -> SplitPaths:
    """
    If you extracted Speech Commands v0.02, it often includes:
      validation_list.txt
      testing_list.txt

    Training is then: all labeled wavs excluding validation+testing lists.
    """
    root = Path(root)
    val = root / "validation_list.txt"
    test = root / "testing_list.txt"
    return SplitPaths(
        train_list=None,
        val_list=val if val.exists() else None,
        test_list=test if test.exists() else None,
    )

