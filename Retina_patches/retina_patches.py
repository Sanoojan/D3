"""
Generate retina-like multi-resolution patches from a video file.

This script reads a video, optionally detects faces/eyes, and crops multi-scale
patches (retina patches) centered on detected landmarks (eyes or face center).
It saves patches to an output directory and can optionally store numpy arrays
(.npz) with patches for downstream processing.

Dependencies:
- OpenCV (cv2)
- numpy

Example usage:
python tasks/sapiensID/retina_patches.py --video /egr/research-sprintai/baliahsa/projects/TALL4Deepfake/data/DeepfakeDatasets/DFDC/test_videos/aassnaulhq.mp4 --outdir patches --scales 1.0 0.7 0.4 --format png --step 5

Options:
--video    Path to input video (required)
--outdir   Directory to write patches (default: ./retina_patches)
--scales   Space-separated scales relative to a base box size. Default: 1.0 0.7 0.4
--bbox-size Base box size in pixels (default: 224). Scales multiply this size.
--detector Which detector to use: 'haar' (default) or 'center'. 'haar' tries to detect eyes/faces.
--step     Process every Nth frame (default: 10)
--format   Output image format: png or npz (default: png)
--verbose  Print progress

Notes:
- Haar cascade XMLs bundled with OpenCV are used for face/eye detection. If not
  present, the script falls back to using the frame center as crop location.
- The script writes one folder per frame and saves patches per scale.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Try to locate Haar cascade files in opencv data
def _get_haar_cascade(name: str) -> Optional[str]:
    # Common cv2 data path
    try:
        base = cv2.data.haarcascades
    except Exception:
        base = None
    if base:
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    # fallback: try relative install path
    script_dir = Path(__file__).parent
    candidate = script_dir / name
    if candidate.exists():
        return str(candidate)
    return None


def detect_landmarks(frame: np.ndarray, detector: str = "haar") -> List[Tuple[int, int]]:
    """Return a list of (x,y) centers to crop around.

    For 'haar' detector, prefer eyes; if not found, use face center. For 'center',
    simply return the frame center.
    """
    h, w = frame.shape[:2]
    centers = []
    if detector == "center":
        centers.append((w // 2, h // 2))
        return centers

    # Haar detector path for eyes and face
    eye_xml = _get_haar_cascade("haarcascade_eye.xml")
    face_xml = _get_haar_cascade("haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = []
    faces = []
    if eye_xml:
        eye_cascade = cv2.CascadeClassifier(eye_xml)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    if len(eyes) > 0:
        # use midpoints of detected eyes (pair them by y coordinate roughly)
        eye_centers = [(int(x + w_ / 2), int(y + h_ / 2)) for (x, y, w_, h_) in eyes]
        # If many eyes, pick two with smallest y-distance pairing
        if len(eye_centers) >= 2:
            # pick the two with left/right ordering by x
            eye_centers = sorted(eye_centers, key=lambda p: p[0])
            left, right = eye_centers[0], eye_centers[1]
            centers.append(((left[0] + right[0]) // 2, (left[1] + right[1]) // 2))
        else:
            centers.extend(eye_centers)
        return centers

    if face_xml:
        face_cascade = cv2.CascadeClassifier(face_xml)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # pick largest face
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w_, h_ = faces[0]
        centers.append((int(x + w_ / 2), int(y + h_ / 2)))
        return centers

    # fallback to center
    centers.append((w // 2, h // 2))
    return centers


def extract_patch(frame: np.ndarray, center: Tuple[int, int], size: int) -> np.ndarray:
    """Extract a square patch centered at `center` with side `size`.
    Pads with border replication if patch goes outside image.
    """
    h, w = frame.shape[:2]
    cx, cy = center
    half = size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w + 1)
    pad_bottom = max(0, y2 - h + 1)

    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(w - 1, x2)
    y2_clamped = min(h - 1, y2)

    patch = frame[y1_clamped:y2_clamped + 1, x1_clamped:x2_clamped + 1]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_REPLICATE)
    # ensure exact size
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return patch


def process_video(
    video_path: str,
    outdir: str,
    scales: List[float],
    base_size: int = 224,
    step: int = 10,
    detector: str = "haar",
    out_format: str = "png",
    verbose: bool = False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        centers = detect_landmarks(frame, detector=detector)
        for ci, center in enumerate(centers):
            frame_folder = Path(outdir) / f"frame_{frame_idx:08d}" / f"center_{ci}"
            frame_folder.mkdir(parents=True, exist_ok=True)
            patches = []
            for s in scales:
                size = max(8, int(round(base_size * s)))
                patch = extract_patch(frame, center, size)
                patches.append(patch)
                if out_format == "png":
                    out_path = frame_folder / f"patch_{int(s*100):03d}_{size}.png"
                    cv2.imwrite(str(out_path), patch)
                # else we will save as npz below
            if out_format == "npz":
                np_path = frame_folder / "patches.npz"
                # store as uint8 RGB
                np.savez_compressed(str(np_path), *patches)
        saved += 1
        if verbose and saved % 10 == 0:
            print(f"Processed frames: {frame_idx} -> saved {saved} frame folders")
        frame_idx += 1

    cap.release()
    if verbose:
        print(f"Done. Frames processed (sampled): {frame_idx // step}. Output -> {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate retina-like patches from a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--outdir", default="./retina_patches", help="Output directory")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.7, 0.4],
                        help="Scales relative to base box size")
    parser.add_argument("--bbox-size", type=int, default=224, dest="base_size",
                        help="Base box size in pixels")
    parser.add_argument("--detector", choices=["haar", "center"], default="haar",
                        help="Which detector to use for centers")
    parser.add_argument("--step", type=int, default=10, help="Process every Nth frame")
    parser.add_argument("--format", choices=["png", "npz"], default="png", dest="out_format",
                        help="Output file format for patches")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.video, args.outdir, args.scales, base_size=args.base_size,
                  step=args.step, detector=args.detector, out_format=args.out_format, verbose=args.verbose)
