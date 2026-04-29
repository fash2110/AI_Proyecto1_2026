from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import torch

from src.datasets.speech_commands import build_official_split_indices
from src.preprocess.config import AudioPreprocessConfig
from src.preprocess.litert_parity import checklist_from_cfg
from src.preprocess.pipeline import AugmentConfig, wav_path_to_input_tensor


def _safe_stem_for_wav(root: Path, wav_path: Path) -> str:
    rel = str(wav_path.relative_to(root)).replace(os.sep, "/")
    h = sha1(rel.encode("utf-8")).hexdigest()[:12]
    rel_safe = rel.replace("/", "__").replace("..", "_")
    rel_safe = rel_safe[:-4] if rel_safe.lower().endswith(".wav") else rel_safe
    return f"{rel_safe}__{h}"


def _save_tensor(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), path)


def _fallback_stratified_split(
    items: list[tuple[Path, int]],
    *,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, list[tuple[Path, int]]]:
    """
    Deterministic per-class split when official split lists are missing.
    """
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for p, y in items:
        by_label.setdefault(int(y), []).append((p, int(y)))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    out: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for y, lst in by_label.items():
        n = len(lst)
        if n == 0:
            continue
        perm = torch.randperm(n, generator=g).tolist()
        lst = [lst[i] for i in perm]

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val
        if n_test == 0 and n_val > 0:
            n_val -= 1
            n_test += 1

        out["train"].extend(lst[:n_train])
        out["val"].extend(lst[n_train : n_train + n_val])
        out["test"].extend(lst[n_train + n_val :])

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="./Dataset", help="Speech Commands root folder")
    ap.add_argument("--out", type=str, default="./cache_mels", help="Output cache folder")
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N per split (sanity runs)")
    ap.add_argument("--seed", type=int, default=1337, help="Base seed for deterministic augmentation")
    ap.add_argument("--log_every", type=int, default=1000, help="Print progress every N items")
    ap.add_argument("--split", type=str, default="", help="If set, process only one split: train|val|test")
    ap.add_argument("--start", type=int, default=0, help="Start index within the split (for chunked runs)")
    ap.add_argument("--count", type=int, default=0, help="How many items to process from start (0 means all)")
    ap.add_argument("--append_manifest", action="store_true", help="Append to existing manifest.jsonl")
    ap.add_argument("--skip_existing", action="store_true", help="Skip items whose cache files already exist")
    ap.add_argument(
        "--fallback_split",
        action="store_true",
        help="If official split lists are missing, do a deterministic stratified split.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = AudioPreprocessConfig()
    parity = checklist_from_cfg(cfg)
    (out_root / "preprocess_contract.json").write_text(
        json.dumps(asdict(parity), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    splits = build_official_split_indices(dataset_root)
    # If dataset doesn't include validation/testing lists, val/test are empty.
    if args.fallback_split and (len(splits["val"].items) == 0 and len(splits["test"].items) == 0):
        all_items = splits["train"].items
        split_items = _fallback_stratified_split(all_items, seed=int(args.seed))
        splits = {
            "train": type(splits["train"])(  # SpeechCommandsIndex
                items=split_items["train"],
                label_to_idx=splits["train"].label_to_idx,
                idx_to_label=splits["train"].idx_to_label,
            ),
            "val": type(splits["train"])(
                items=split_items["val"],
                label_to_idx=splits["train"].label_to_idx,
                idx_to_label=splits["train"].idx_to_label,
            ),
            "test": type(splits["train"])(
                items=split_items["test"],
                label_to_idx=splits["train"].label_to_idx,
                idx_to_label=splits["train"].idx_to_label,
            ),
        }

    raw_aug = AugmentConfig(enabled=False)
    any_aug = AugmentConfig(enabled=True)

    manifest_path = out_root / "manifest.jsonl"
    f_manifest = manifest_path.open("a" if args.append_manifest else "w", encoding="utf-8")

    t0 = time.time()
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    per_class: dict[str, dict[int, int]] = {"train": {}, "val": {}, "test": {}}
    seen_shapes: list[tuple[int, int, int]] = []

    split_names = ("train", "val", "test")
    if args.split:
        if args.split not in split_names:
            raise ValueError("--split must be one of: train, val, test")
        split_names = (args.split,)

    for split_name in split_names:
        items = splits[split_name].items
        if args.limit and args.limit > 0:
            items = items[: int(args.limit)]
        if args.start and args.start > 0:
            items = items[int(args.start) :]
        if args.count and args.count > 0:
            items = items[: int(args.count)]

        for i, (wav_path, label_idx) in enumerate(items):
            rng = torch.Generator(device="cpu")
            global_i = int(args.start) + i
            rng.manual_seed(int(args.seed) + global_i)

            stem = _safe_stem_for_wav(dataset_root, wav_path)

            # Raw
            raw_path = out_root / "raw" / split_name / f"{stem}.pt"
            aug_path = out_root / "aug" / split_name / f"{stem}.pt"

            if args.skip_existing and raw_path.exists() and aug_path.exists():
                x_raw = torch.load(raw_path, map_location="cpu")
            else:
                x_raw = wav_path_to_input_tensor(wav_path, cfg, augment=raw_aug, rng=rng)
                _save_tensor(raw_path, x_raw)

                # Augmented (train/val/test as requested)
                rng_aug = torch.Generator(device="cpu")
                rng_aug.manual_seed(int(args.seed) + global_i)  # deterministic per item
                x_aug = wav_path_to_input_tensor(wav_path, cfg, augment=any_aug, rng=rng_aug)
                _save_tensor(aug_path, x_aug)

            shape = tuple(int(s) for s in x_raw.shape)  # type: ignore[assignment]
            if len(seen_shapes) < 5:
                seen_shapes.append(shape)  # sample a few

            counts[split_name] += 1
            per_class[split_name][int(label_idx)] = per_class[split_name].get(int(label_idx), 0) + 1

            record = {
                "split": split_name,
                "wav_relpath": str(wav_path.relative_to(dataset_root)).replace(os.sep, "/"),
                "label_idx": int(label_idx),
                "label": splits[split_name].idx_to_label[int(label_idx)],
                "cache_path_raw": str(raw_path.relative_to(out_root)).replace(os.sep, "/"),
                "cache_path_aug": str(aug_path.relative_to(out_root)).replace(os.sep, "/"),
                "shape": list(shape),
            }
            f_manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.log_every and (i + 1) % int(args.log_every) == 0:
                f_manifest.flush()
                print(f"[{split_name}] processed {i+1}/{len(items)}", flush=True)

    f_manifest.close()
    elapsed = time.time() - t0

    print("Done.")
    print("dataset_root:", dataset_root)
    print("out_root:", out_root)
    print("counts:", counts)
    print("per_class_counts:", per_class)
    print("example_shapes:", seen_shapes)
    print(f"elapsed_s: {elapsed:.2f}")


if __name__ == "__main__":
    main()

