"""
Utility to delete a single trial (episode) and all associated data
from a LeRobot-formatted dataset.

Intended usage:

    python -m franka_table.datasets.delete_trial \\
        --dataset_root franka_table/datasets/franka_table_synth \\
        --trial_index 1

This will remove:
  - data/chunk-000/episode_{episode_index:06d}.parquet
  - the corresponding row from meta/episodes/chunk-000/file-000.parquet
  - any lines for that episode from episodes.jsonl (if present)
  - any MP4 shards whose filenames start with trial_{episode_index:03d}_
    or match file-{episode_index:03d}.mp4 under videos/{camera}/chunk-000/

Note: meta/stats.json is NOT recomputed by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def delete_episode_parquet(dataset_root: Path, episode_index: int) -> None:
    data_dir = dataset_root / "data" / "chunk-000"
    pq_path = data_dir / f"episode_{episode_index:06d}.parquet"
    if pq_path.exists():
        pq_path.unlink()
        print(f"Deleted data shard: {pq_path}")
    else:
        print(f"[warn] Data shard not found: {pq_path}")


def update_meta_episodes(dataset_root: Path, episode_index: int) -> None:
    ep_pq = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_pq.exists():
        print(f"[warn] Episodes parquet not found: {ep_pq}")
        return

    df = pd.read_parquet(ep_pq)
    before = len(df)
    df = df[df["episode_index"] != episode_index]
    after = len(df)
    if after == before:
        print(f"[warn] No episode_index {episode_index} found in {ep_pq}")
    else:
        if after == 0:
            ep_pq.unlink()
            print(f"Removed last episode; deleted {ep_pq}")
        else:
            df.to_parquet(ep_pq, index=False)
            print(f"Updated episodes parquet: removed episode_index {episode_index}")


def update_episodes_jsonl(dataset_root: Path, episode_index: int) -> None:
    jsonl_path = dataset_root / "episodes.jsonl"
    if not jsonl_path.exists():
        return

    lines: List[str] = []
    removed = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("episode_index") == episode_index:
                removed += 1
                continue
            lines.append(line)

    if removed == 0:
        print(f"[warn] No episode_index {episode_index} found in {jsonl_path}")
        return

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
    print(f"Updated episodes.jsonl: removed {removed} record(s) for episode_index {episode_index}")


def delete_video_shards(
    dataset_root: Path,
    video_paths: Dict[str, str] | None = None,
    cameras: List[str] | None = None,
) -> None:
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        print(f"[info] No videos/ directory under {dataset_root}, skipping video deletion")
        return

    # Delete explicit video paths first (from mapping)
    if video_paths:
        for cam, rel_path in video_paths.items():
            abs_path = dataset_root / rel_path
            if abs_path.exists():
                abs_path.unlink()
                print(f"Deleted video shard (from mapping): {abs_path}")

    # Optionally, also scan by camera for any remaining shards (no extra heuristics for now)
    if cameras:
        cams = cameras
    else:
        cams = [p.name for p in videos_root.iterdir() if p.is_dir()]
    for cam in cams:
        cam_chunk = videos_root / cam / "chunk-000"
        if not cam_chunk.exists():
            continue
        # No additional pattern-based deletion; mapping is the primary source


def resolve_episode_from_trial(
    dataset_root: Path, trial_index: int
) -> Tuple[int, Dict[str, str]]:
    """Resolve episode_index and video paths from meta/trial_mapping.jsonl for a given trial_index."""
    mapping_path = dataset_root / "meta" / "trial_mapping.jsonl"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Trial mapping file not found: {mapping_path}")

    chosen: Dict[str, any] | None = None
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("trial_index") == trial_index:
                chosen = rec  # keep last match

    if chosen is None:
        raise ValueError(f"No mapping found for trial_index={trial_index} in {mapping_path}")

    ep_idx = int(chosen["episode_index"])
    videos = chosen.get("videos", {}) or {}
    return ep_idx, videos


def update_trial_mapping(dataset_root: Path, trial_index: int) -> None:
    """Remove entries for a given trial_index from meta/trial_mapping.jsonl."""
    mapping_path = dataset_root / "meta" / "trial_mapping.jsonl"
    if not mapping_path.exists():
        return

    kept: List[str] = []
    removed = 0
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("trial_index") == trial_index:
                removed += 1
                continue
            kept.append(line)

    if removed == 0:
        print(f"[warn] No trial_index {trial_index} found in {mapping_path}")
        return

    with open(mapping_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line)
    print(f"Updated trial_mapping.jsonl: removed {removed} record(s) for trial_index {trial_index}")


def recompute_info_and_stats(dataset_root: Path) -> None:
    """Recompute meta/info.json counts and meta/stats.json from current shards."""
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_parquet = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"

    if not info_path.exists():
        print(f"[warn] info.json not found at {info_path}, skipping info/stats recompute")
        return

    # Load existing info to preserve schema and paths
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Episodes and frame counts
    n_episodes = 0
    n_frames = 0
    if episodes_parquet.exists():
        ep_df = pd.read_parquet(episodes_parquet)
        if "episode_index" in ep_df.columns:
            n_episodes = int(ep_df["episode_index"].nunique())
        if "length" in ep_df.columns:
            n_frames = int(ep_df["length"].sum())

    # Tasks count from tasks parquet if present
    tasks_parquet = meta_dir / "tasks.parquet"
    n_tasks = 0
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        if "task_index" in tasks_df.columns:
            n_tasks = int(tasks_df["task_index"].nunique())

    info["total_episodes"] = n_episodes
    info["total_frames"] = n_frames
    info["total_tasks"] = n_tasks

    # Recompute stats for state/actions from all data shards
    stats: Dict[str, Dict[str, List[float]]] = {}
    data_dir = dataset_root / "data" / "chunk-000"
    if data_dir.exists():
        all_states: List = []
        all_actions: List = []
        for pq_path in sorted(data_dir.glob("episode_*.parquet")):
            table = pd.read_parquet(pq_path)
            if "observation.state" in table.columns and len(table) > 0:
                states_ep = table["observation.state"].to_list()
                all_states.extend(states_ep)
            if "action" in table.columns and len(table) > 0:
                actions_ep = table["action"].to_list()
                all_actions.extend(actions_ep)

        if all_states:
            import numpy as np

            states_arr = np.asarray(all_states, dtype=np.float32)
            stats["state"] = {
                "mean": states_arr.mean(axis=0).tolist(),
                "std": states_arr.std(axis=0).tolist(),
                "min": states_arr.min(axis=0).tolist(),
                "max": states_arr.max(axis=0).tolist(),
            }
        if all_actions:
            import numpy as np

            actions_arr = np.asarray(all_actions, dtype=np.float32)
            stats["actions"] = {
                "mean": actions_arr.mean(axis=0).tolist(),
                "std": actions_arr.std(axis=0).tolist(),
                "min": actions_arr.min(axis=0).tolist(),
                "max": actions_arr.max(axis=0).tolist(),
            }

    # Write updated info
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Write updated stats
    stats_path = meta_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Recomputed info.json and stats.json (episodes={n_episodes}, frames={n_frames}, tasks={n_tasks})")

def main() -> None:
    parser = argparse.ArgumentParser(description="Delete a trial/episode from a LeRobot-formatted dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset root (e.g., franka_table/datasets/franka_table_synth)",
    )
    parser.add_argument(
        "--trial_index",
        type=int,
        required=True,
        help="Trial index to delete (as used in the sweep script).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of camera names whose videos should be cleaned (defaults to all subdirs of videos/).",
    )

    args = parser.parse_args()
    root = Path(args.dataset_root)
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    trial_idx = int(args.trial_index)
    ep_idx, video_paths = resolve_episode_from_trial(root, trial_idx)
    print(f"Deleting trial_index={trial_idx}, episode_index={ep_idx} from dataset at {root}")

    delete_episode_parquet(root, ep_idx)
    update_meta_episodes(root, ep_idx)
    update_episodes_jsonl(root, ep_idx)
    delete_video_shards(root, video_paths=video_paths, cameras=args.cameras)
    update_trial_mapping(root, trial_idx)
    recompute_info_and_stats(root)

    print("Done. info.json, stats.json, and trial_mapping.jsonl updated.")


if __name__ == "__main__":
    main()
