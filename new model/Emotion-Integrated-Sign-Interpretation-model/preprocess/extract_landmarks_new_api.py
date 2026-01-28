"""Extract MediaPipe landmarks from videos using new API (mediapipe 0.10.x)."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

# Constants for landmark counts
FACE_POINTS = 468
HAND_POINTS = 21
POSE_POINTS = 33

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("extract_new")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract landmarks using MediaPipe new API."
    )
    parser.add_argument("video_dir", type=Path, help="Directory containing videos.")
    parser.add_argument("output_dir", type=Path, help="Where to store .npz landmarks.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="CSV manifest containing metadata for each sample.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=48,
        help="Number of frames per sequence after padding/cropping.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of parallel workers to use."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Safety limit on frames processed per clip.",
    )
    return parser.parse_args()


def pad_or_crop(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or crop array to target length."""
    if len(arr) == target_len:
        return arr
    elif len(arr) < target_len:
        return np.pad(arr, ((0, target_len - len(arr)), (0, 0)), mode="constant")
    else:
        return arr[:target_len]


def extract_video(
    video_path: Path, seq_len: int, max_frames: int = 300
) -> Dict[str, np.ndarray]:
    """Extract landmarks from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    frames_hand_left: List[np.ndarray] = []
    frames_hand_right: List[np.ndarray] = []
    frames_face: List[np.ndarray] = []
    frames_pose: List[np.ndarray] = []
    frame_count = 0

    # Initialize MediaPipe Vision tasks
    base_options = mp.tasks.BaseOptions(
        model_asset_path="model.tflite", delegate=mp.tasks.BaseOptions.delegate.CPU
    )

    pose_options = mp.tasks.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
    )

    hand_options = mp.tasks.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
    )

    face_options = mp.tasks.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        output_facial_landmark_confidence=False,
    )

    with (
        mp.tasks.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        mp.tasks.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        mp.tasks.FaceLandmarker.create_from_options(face_options) as face_landmarker,
    ):
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Extract landmarks
            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            # Process hand landmarks
            if hand_result and hand_result.handedness:
                # Determine left/right hands
                left_landmarks = np.zeros((HAND_POINTS, 3), dtype=np.float32)
                right_landmarks = np.zeros((HAND_POINTS, 3), dtype=np.float32)

                for i, handedness in enumerate(hand_result.handedness):
                    if handedness == "Left" and i < len(hand_result.hand_landmarks):
                        lm = hand_result.hand_landmarks[i]
                        for j in range(min(HAND_POINTS, len(lm))):
                            left_landmarks[j] = [lm.x, lm.y, lm.z]
                    elif handedness == "Right" and i < len(hand_result.hand_landmarks):
                        lm = hand_result.hand_landmarks[i]
                        for j in range(min(HAND_POINTS, len(lm))):
                            right_landmarks[j] = [lm.x, lm.y, lm.z]

                frames_hand_left.append(left_landmarks)
                frames_hand_right.append(right_landmarks)
            else:
                frames_hand_left.append(np.zeros((HAND_POINTS, 3), dtype=np.float32))
                frames_hand_right.append(np.zeros((HAND_POINTS, 3), dtype=np.float32))

            # Process face landmarks
            if face_result and face_result.face_landmarks:
                face_lm = face_result.face_landmarks[0]
                face_landmarks_array = np.zeros((FACE_POINTS, 3), dtype=np.float32)
                for i in range(min(FACE_POINTS, len(face_lm))):
                    face_landmarks_array[i] = [face_lm.x, face_lm.y, face_lm.z]
                frames_face.append(face_landmarks_array)
            else:
                frames_face.append(np.zeros((FACE_POINTS, 3), dtype=np.float32))

            # Process pose landmarks
            if pose_result and pose_result.pose_landmarks:
                pose_lm = pose_result.pose_landmarks[0]
                pose_landmarks_array = np.zeros((POSE_POINTS, 3), dtype=np.float32)
                for i in range(min(POSE_POINTS, len(pose_lm))):
                    pose_landmarks_array[i] = [pose_lm.x, pose_lm.y, pose_lm.z]
                frames_pose.append(pose_landmarks_array)
            else:
                frames_pose.append(np.zeros((POSE_POINTS, 3), dtype=np.float32))

            frame_count += 1
            if frame_count >= max_frames:
                LOGGER.warning(
                    "Reached frame limit (%s) for %s; truncating clip",
                    max_frames,
                    video_path.name,
                )
                break

    cap.release()

    if not frames_hand_left:
        raise ValueError(f"No frames decoded from {video_path}")

    sample = {
        "hand_left": pad_or_crop(np.stack(frames_hand_left), seq_len),
        "hand_right": pad_or_crop(np.stack(frames_hand_right), seq_len),
        "face": pad_or_crop(np.stack(frames_face), seq_len),
        "pose": pad_or_crop(np.stack(frames_pose), seq_len),
    }
    return sample


def _process_video_file(
    video_path: Path,
    seq_len: int,
    max_frames: int,
    output_dir: Path,
) -> Optional[Path]:
    """Process a single video file and save landmarks."""
    cv2.setNumThreads(0)
    try:
        sample = extract_video(video_path, seq_len, max_frames)

        # Create output path: output_dir/word/filename.npz
        stem = video_path.stem
        parts = stem.split("__")
        if len(parts) >= 4:
            word = "__".join(parts[:-4]) if len(parts) > 4 else parts[-5]
            word_dir = output_dir / word
            word_dir.mkdir(parents=True, exist_ok=True)
            output_path = word_dir / f"{stem}.npz"

            np.savez_compressed(output_path, **sample)

            metadata = {
                "source": str(video_path),
                "sequence_length": seq_len,
            }
            (output_path.with_suffix(".json")).write_text(json.dumps(metadata))

            return output_path
        else:
            LOGGER.warning("Skipping %s: invalid filename format", video_path.name)
            return None
    except Exception as e:
        LOGGER.exception("Failed to process %s: %s", video_path, e)
        return None


def main() -> None:
    args = parse_args()
    LOGGER.info("Extracting landmarks using MediaPipe 0.10.x new API")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(args.video_dir.glob("*.mp4"))
    LOGGER.info("Processing %d videos", len(video_paths))

    if args.num_workers <= 1:
        for video_path in video_paths:
            LOGGER.info("Processing %s", video_path.name)
            output_path = _process_video_file(
                video_path, args.sequence_length, args.max_frames, args.output_dir
            )
            if output_path:
                LOGGER.info("Saved %s", output_path)
    else:
        LOGGER.info(
            "Processing %d videos with %d workers", len(video_paths), args.num_workers
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    _process_video_file,
                    video_path,
                    args.sequence_length,
                    args.max_frames,
                    args.output_dir,
                )
                for video_path in video_paths
            ]
            for future in futures:
                output_path = future.result()
                if output_path:
                    LOGGER.info("Saved %s", output_path)

    LOGGER.info("Landmark extraction complete!")


if __name__ == "__main__":
    main()
