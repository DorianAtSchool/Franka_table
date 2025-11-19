import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import imageio.v2 as imageio


def color_jitter(img: np.ndarray, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    # Brightness
    if brightness > 0:
        b = (np.random.rand() * 2 - 1) * brightness
        x = np.clip(x + b, 0.0, 1.0)
    # Contrast
    if contrast > 0:
        c = 1.0 + (np.random.rand() * 2 - 1) * contrast
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = np.clip((x - mean) * c + mean, 0.0, 1.0)
    # Saturation & hue in HSV
    if saturation > 0 or hue > 0:
        import cv2
        hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        if saturation > 0:
            s_scale = 1.0 + (np.random.rand() * 2 - 1) * saturation
            hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
        if hue > 0:
            h_shift = int((np.random.rand() * 2 - 1) * hue * 180)
            hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return (x * 255.0).astype(np.uint8)


def add_noise(img: np.ndarray, sigma=5.0) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    x = np.clip(img.astype(np.float32) + noise, 0, 255)
    return x.astype(np.uint8)


def augment_frame(img: np.ndarray) -> np.ndarray:
    out = color_jitter(img, brightness=0.2, contrast=0.25, saturation=0.2, hue=0.03)
    if np.random.rand() < 0.5:
        out = add_noise(out, sigma=3.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Augment an existing LeRobot-like dataset by image-only synthetic variations.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--copies", type=int, default=3, help="Number of augmented copies per episode")
    parser.add_argument("--suffix", type=str, default="aug", help="Suffix for augmented episodes")
    args = parser.parse_args()

    pkg_root = Path(__file__).resolve().parents[1]
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = pkg_root / root
    meta_dir = root / "meta"
    frames_dir = root / "frames"
    data_dir = root / "data" / "chunk-000"
    episodes_file = root / "episodes.jsonl"
    info_file = meta_dir / "info.json"

    if not episodes_file.exists():
        raise FileNotFoundError(f"Missing {episodes_file}")
    if not info_file.exists():
        raise FileNotFoundError(f"Missing {info_file}")

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    cameras = info.get("cameras", [])

    # Load existing episodes
    episodes: List[dict] = []
    with open(episodes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    next_episode_index = max((ep["episode_index"] for ep in episodes), default=-1) + 1

    new_episodes: List[dict] = []
    for ep in episodes:
        parquet_rel = ep["parquet"]
        parquet_path = root / parquet_rel
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        for copy_idx in range(args.copies):
            new_ep_idx = next_episode_index
            next_episode_index += 1
            ep_prefix = f"episode_{new_ep_idx:06d}"
            out_parquet = data_dir / f"{ep_prefix}.parquet"

            # Prepare new rows
            new_rows = []
            for i, row in df.iterrows():
                new_row = row.to_dict()
                # Update episode and frame identifiers
                new_row["episode_index"] = new_ep_idx
                new_row["index"] = int(i)

                # Augment images and store new paths
                for cam in cameras:
                    col = f"observation.images.{cam}"
                    if col in new_row and isinstance(new_row[col], str):
                        in_rel = new_row[col]
                        in_path = frames_dir / in_rel
                        img = imageio.imread(in_path)
                        aug = augment_frame(img)
                        # New path under frames/<cam>/<ep_prefix>_aug/frame_xxx.png
                        out_dir = frames_dir / cam / (ep_prefix + f"_{args.suffix}")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / Path(in_path).name
                        imageio.imwrite(out_path, aug)
                        rel_out = os.path.relpath(out_path, frames_dir).replace(os.sep, "/")
                        new_row[col] = rel_out

                new_rows.append(new_row)

            out_df = pd.DataFrame(new_rows)
            out_df.to_parquet(out_parquet, index=False)

            new_episodes.append({
                "episode_index": new_ep_idx,
                "length": int(len(new_rows)),
                "tasks": ep.get("tasks", []),
                "parquet": os.path.relpath(out_parquet, root).replace(os.sep, "/"),
                "cameras": cameras,
            })

    # Append new episodes
    with open(episodes_file, "a", encoding="utf-8") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # Update info.json counts
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    info["num_episodes"] = int(info.get("num_episodes", 0) + len(new_episodes))
    info["num_frames"] = int(info.get("num_frames", 0) + sum(ep["length"] for ep in new_episodes))
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Augmented {len(new_episodes)} episodes. Dataset updated at {root}")


if __name__ == "__main__":
    main()
