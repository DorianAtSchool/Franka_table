import argparse
import json
from pathlib import Path
from typing import List, Set, Optional

import imageio.v2 as imageio
import numpy as np
import cv2


def load_episodes(root: Path) -> List[dict]:
    eps: List[dict] = []
    episodes_file = root / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Missing {episodes_file}")
    with open(episodes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eps.append(json.loads(line))
    return eps


def load_cameras(root: Path) -> List[str]:
    info_file = root / "meta" / "info.json"
    cameras: List[str] = []
    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            cams = info.get("cameras", [])
            if isinstance(cams, list):
                cameras = cams
        except Exception:
            pass
    if not cameras:
        frames_dir = root / "frames"
        if frames_dir.exists():
            cameras = sorted([p.name for p in frames_dir.iterdir() if p.is_dir()])
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Export a dataset episode to an MP4 by stitching saved frames (no re-simulation).")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--episode_index", type=int, required=True, help="Episode index to export")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path (default: dataset_root/video_episode_XXXXXX.mp4)")
    parser.add_argument("--fps", type=int, default=None, help="FPS for the video (default: meta fps or 25)")
    parser.add_argument("--cameras", type=str, nargs="*", default=None, help="Subset of cameras to include (default: all)")
    parser.add_argument("--height", type=int, default=360, help="Target tile height (preserve aspect with letterboxing)")
    parser.add_argument("--cols", type=int, default=0, help="Grid columns (0=auto)")
    parser.add_argument("--gutter", type=int, default=8, help="Padding between tiles (pixels)")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root

    episodes = load_episodes(root)
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"episode_index {args.episode_index} out of range (0..{len(episodes)-1})")
    ep = episodes[args.episode_index]

    # Resolve fps
    fps = args.fps or 25
    info_file = root / "meta" / "info.json"
    if info_file.exists() and args.fps is None:
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            fps = int(info.get("fps", fps))
        except Exception:
            pass

    # Resolve cameras
    cameras_all = load_cameras(root)
    cameras_active: Set[str] = set(args.cameras) if args.cameras else set(cameras_all)
    if not cameras_active:
        raise ValueError("No cameras available to export")

    frames_dir = root / "frames"
    prefix = Path(ep["parquet"]).stem  # episode_XXXXXX
    length = int(ep.get("length", 0))
    if length <= 0:
        # Fallback: count frames in first active camera folder
        cam = next(iter(cameras_active))
        frame_folder = frames_dir / cam / prefix
        length = len(list(frame_folder.glob("frame_*.png")))

    # Grid layout
    n = len(cameras_active)
    cols = args.cols if args.cols and args.cols > 0 else int(np.ceil(np.sqrt(n)))
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    target_h = max(1, int(args.height))
    gutter = max(0, int(args.gutter))

    # Determine average aspect to choose nominal tile width
    aspects = []
    for cam in cameras_active:
        # read first available frame to estimate aspect
        p0 = frames_dir / cam / prefix / f"frame_{0:06d}.png"
        if p0.exists():
            try:
                im0 = imageio.imread(p0)
                aspects.append(im0.shape[1] / im0.shape[0])
            except Exception:
                pass
    avg_aspect = float(np.mean(aspects)) if aspects else (16.0 / 9.0)
    target_w = int(round(target_h * avg_aspect))

    # Build writer
    mosaic_h = rows * target_h + (rows - 1) * gutter
    mosaic_w = cols * target_w + (cols - 1) * gutter
    out_path = Path(args.out) if args.out else (root / f"video_{prefix}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)

    # Helper: letterbox resize to fit tile while preserving aspect
    def fit_letterbox(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(tw / float(w), th / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh))
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = resized
        return out

    cams_order = list(cameras_active)

    for t in range(length):
        canvas = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        for idx, cam in enumerate(cams_order):
            p = frames_dir / cam / prefix / f"frame_{t:06d}.png"
            if not p.exists():
                continue
            try:
                img = imageio.imread(p)
            except Exception:
                continue
            tile = fit_letterbox(img, target_w, target_h)
            r = idx // cols
            c = idx % cols
            x = c * (target_w + gutter)
            y = r * (target_h + gutter)
            canvas[y:y + target_h, x:x + target_w] = tile
        writer.append_data(canvas)

    writer.close()
    print(f"Wrote {out_path} ({mosaic_w}x{mosaic_h} @ {fps} FPS, {length} frames)")


if __name__ == "__main__":
    main()

