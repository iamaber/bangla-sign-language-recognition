"""Real-time BdSL recognition demo (without AI tutor)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.constants import FACE_POINTS, HAND_POINTS, POSE_POINTS
from models.fusion import FusionModel
from preprocess.normalize import NormalizationConfig, normalize_sample
from train.vocab import build_vocab_from_manifest

GRAMMAR_IDX_TO_TAG = ["neutral", "question", "negation", "happy", "sad"]
DEFAULT_STABLE_FRAMES = 10
DEFAULT_MIN_CONF = 0.60
MAX_SENTENCE_WORDS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time BdSL recognition demo.")
    parser.add_argument(
        "checkpoint", type=Path, help="Path to trained fusion model weights."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--buffer", type=int, default=48, help="Sliding window length for model input."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Manifest CSV to recover vocabulary labels.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=DEFAULT_STABLE_FRAMES,
        help="Frames required before accepting a word.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=DEFAULT_MIN_CONF,
        help="Confidence threshold for stable word selection.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("demo/kalpurush.ttf"),
        help="Path to Bangla font (kalpurush/SolaimanLipi).",
    )
    return parser.parse_args()


def load_labels(manifest_path: Path) -> list[str]:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return []
    try:
        vocab = build_vocab_from_manifest(manifest_path)
    except Exception:
        return []
    return vocab.idx_to_label


def _landmark_array(landmarks, size: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((size, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _init_buffers(size: int) -> dict[str, dict[str, np.ndarray | int]]:
    return {
        "hand_left": {
            "data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32),
            "write_idx": 0,
            "filled": 0,
        },
        "hand_right": {
            "data": np.zeros((size, HAND_POINTS, 3), dtype=np.float32),
            "write_idx": 0,
            "filled": 0,
        },
        "face": {
            "data": np.zeros((size, FACE_POINTS, 3), dtype=np.float32),
            "write_idx": 0,
            "filled": 0,
        },
        "pose": {
            "data": np.zeros((size, POSE_POINTS, 3), dtype=np.float32),
            "write_idx": 0,
            "filled": 0,
        },
    }


def _append_sample(
    buffers: dict[str, dict[str, np.ndarray | int]], sample: dict[str, np.ndarray]
) -> None:
    for key, buffer in buffers.items():
        buffer["data"][buffer["write_idx"]] = sample[key]
    first = next(iter(buffers.values()))
    first["write_idx"] = (first["write_idx"] + 1) % first["data"].shape[0]
    first["filled"] = min(first["filled"] + 1, first["data"].shape[0])
    for buffer in buffers.values():
        buffer["write_idx"] = first["write_idx"]
        buffer["filled"] = first["filled"]


def _is_full(buffers: dict[str, dict[str, np.ndarray | int]]) -> bool:
    meta = next(iter(buffers.values()))
    return meta["filled"] == meta["data"].shape[0]


def _stack_window(
    buffers: dict[str, dict[str, np.ndarray | int]],
) -> dict[str, np.ndarray]:
    stacked = {}
    sample_meta = next(iter(buffers.values()))
    size = sample_meta["data"].shape[0]
    write_idx = sample_meta["write_idx"]
    for key, buffer in buffers.items():
        if buffer["filled"] < size:
            stacked[key] = buffer["data"][: buffer["filled"]]
            continue
        if write_idx == 0:
            stacked[key] = buffer["data"]
        else:
            stacked[key] = np.concatenate(
                (buffer["data"][write_idx:], buffer["data"][:write_idx]), axis=0
            )
    return stacked


def _format_word(sign_idx: int, labels: list[str]) -> str:
    if sign_idx < 0:
        return "..."
    if labels and 0 <= sign_idx < len(labels):
        return labels[sign_idx]
    return f"#{sign_idx}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    holistic = mp.solutions.holistic.Holistic()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        cap.release()
        holistic.close()
        return

    # Initialize font for Bangla text
    font = None
    if args.font_path.exists():
        try:
            font = cv2.freetype.Font(str(args.font_path), 0, 0, 0, 0, 1)
        except Exception:
            font = None

    buffers = _init_buffers(args.buffer)
    config = NormalizationConfig(sequence_length=args.buffer)
    ema_sign = None
    ema_grammar = None
    alpha = 0.6
    idx_to_label = load_labels(args.manifest)
    stable_count = 0
    last_word_frame: Optional[str] = None
    last_stable_word: Optional[str] = None
    sentence_buffer: list[str] = []

    print("Starting real-time sign language recognition...")
    print(f"Device: {device}")
    print(f"Model: {args.checkpoint}")
    print(f"Buffer size: {args.buffer}")
    print(f"Vocabulary size: {len(idx_to_label)} words")
    print("\nControls:")
    print("  'c' - Clear sentence")
    print("  'q' or ESC - Quit")

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(image_rgb)
            sample = {
                "hand_left": _landmark_array(result.left_hand_landmarks, HAND_POINTS),
                "hand_right": _landmark_array(result.right_hand_landmarks, HAND_POINTS),
                "face": _landmark_array(result.face_landmarks, FACE_POINTS),
                "pose": _landmark_array(result.pose_landmarks, POSE_POINTS),
            }
            _append_sample(buffers, sample)

            sign_pred = -1
            grammar_pred = -1
            sign_conf = 0.0
            grammar_tag = "neutral"
            if _is_full(buffers):
                ordered = _stack_window(buffers)
                normalized = normalize_sample(ordered, config)
                tensor_sample = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device).float()
                    for k, v in normalized.items()
                }
                with torch.no_grad():
                    sign_logits, grammar_logits = model(tensor_sample)
                sign_prob = torch.softmax(sign_logits, dim=1)
                grammar_prob = torch.softmax(grammar_logits, dim=1)
                ema_sign = (
                    sign_prob
                    if ema_sign is None
                    else alpha * sign_prob + (1 - alpha) * ema_sign
                )
                ema_grammar = (
                    grammar_prob
                    if ema_grammar is None
                    else alpha * grammar_prob + (1 - alpha) * ema_grammar
                )
                sign_pred = int(torch.argmax(ema_sign))
                grammar_pred = int(torch.argmax(ema_grammar))
                sign_conf = (
                    float(ema_sign[0, sign_pred]) if ema_sign is not None else 0.0
                )
                grammar_tag = (
                    GRAMMAR_IDX_TO_TAG[grammar_pred]
                    if 0 <= grammar_pred < len(GRAMMAR_IDX_TO_TAG)
                    else "neutral"
                )

            current_word = _format_word(sign_pred, idx_to_label)
            if current_word == last_word_frame and sign_conf >= args.min_conf:
                stable_count += 1
            else:
                stable_count = 0
            last_word_frame = current_word

            new_word_added = False
            if stable_count >= args.stable_frames and current_word not in (None, "..."):
                if current_word != last_stable_word:
                    sentence_buffer.append(current_word)
                    last_stable_word = current_word
                    stable_count = 0
                    new_word_added = True
                    if len(sentence_buffer) > MAX_SENTENCE_WORDS:
                        sentence_buffer = sentence_buffer[-MAX_SENTENCE_WORDS:]

            display_sentence = " ".join(sentence_buffer)

            # Prepare overlay frame
            overlay_frame = frame.copy()

            # Display predicted word
            word_text = f"Word: {current_word}"
            if font:
                cv2.putText(
                    overlay_frame,
                    word_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    overlay_frame,
                    word_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Display confidence
            conf_text = f"Conf: {sign_conf:.2f}"
            cv2.putText(
                overlay_frame,
                conf_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Display grammar tag
            tag_color = (0, 255, 0)  # Green for neutral
            if grammar_tag == "question":
                tag_color = (255, 0, 0)  # Blue
            elif grammar_tag in ["negation", "happy", "sad"]:
                tag_color = (0, 0, 255)  # Red
            tag_text = f"Grammar: {grammar_tag}"
            cv2.putText(
                overlay_frame,
                tag_text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tag_color,
                2,
            )

            # Display sentence
            sentence_display = (
                display_sentence
                if grammar_tag != "question"
                else f"{display_sentence}?"
            )
            sent_text = f"Sentence: {sentence_display}"
            if font:
                cv2.putText(
                    overlay_frame,
                    sent_text,
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    overlay_frame,
                    sent_text,
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Display FPS
            fps_val = 1.0 / max((time.time() - start), 1e-6)
            fps_text = f"FPS: {fps_val:.1f}"
            cv2.putText(
                overlay_frame,
                fps_text,
                (10, frame.shape[0] - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Display controls hint
            hint_text = "'c' Clear | 'q' Quit"
            cv2.putText(
                overlay_frame,
                hint_text,
                (10, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                2,
            )

            cv2.imshow("BdSL Recognition", overlay_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                sentence_buffer.clear()
                last_stable_word = None
                stable_count = 0
                print(f"  Cleared sentence")

            if new_word_added:
                print(f"  Added: {current_word}")

    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
