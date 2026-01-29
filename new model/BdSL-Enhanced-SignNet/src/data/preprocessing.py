"""
Data Preprocessing and Augmentation Pipeline for Sign Language Recognition
===========================================================================

Comprehensive data processing for Bengali Sign Language video/pose data.
Includes:
- MediaPipe landmark extraction (body, hands, face)
- Multi-stream normalization
- Temporal augmentation
- Spatial augmentation
- Semantic augmentation

Author: BDSL Recognition Team
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random


# Landmark indices for MediaPipe
class LandmarkIndices:
    """Indices for MediaPipe pose, hand, and face landmarks."""

    # Pose landmarks (33 total)
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Pose connections for visualization
    POSE_CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
    ]

    # Hand landmarks (21 per hand)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Face landmarks (468 total) - simplified regions
    FACE_OUTLINE = list(range(0, 17))
    LEFT_EYEBROW = list(range(17, 22))
    RIGHT_EYEBROW = list(range(22, 27))
    NOSE = list(range(27, 36))
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    LIPS_OUTER = list(range(48, 60))
    LIPS_INNER = list(range(60, 68))


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Dataset paths
    base_dir: str = "/home/raco/Repos/bangla-sign-language-recognition"
    processed_dir: str = "Data/processed/new_model"
    normalized_dir: str = "Data/processed/new_model/normalized"
    checkpoint_dir: str = "Data/processed/new_model/checkpoints"

    # Sequence parameters
    max_seq_length: int = 150
    min_seq_length: int = 10
    target_fps: int = 30

    # Feature dimensions
    body_dim: int = 99  # 33 landmarks * 3 coordinates
    hand_dim: int = 63  # 21 landmarks * 3 coordinates
    face_dim: int = 1404  # 468 landmarks * 3 coordinates

    # Augmentation
    augmentation: bool = True
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02
    rotation_range: float = 15  # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)
    horizontal_flip_prob: float = 0.5

    # Normalization
    use_shoulder_normalization: bool = True
    use_hand_normalization: bool = True


class PoseNormalizer:
    """Normalize pose landmarks using shoulder-based reference."""

    def __init__(self, min_shoulder_scale: float = 0.1):
        self.min_shoulder_scale = min_shoulder_scale

    def normalize(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize pose sequence using shoulder-based centering and scaling.

        Args:
            pose_sequence: Array of shape (num_frames, num_landmarks, 3)

        Returns:
            Normalized pose sequence
        """
        # Calculate shoulder center
        left_shoulder = pose_sequence[:, LandmarkIndices.LEFT_SHOULDER, :3]
        right_shoulder = pose_sequence[:, LandmarkIndices.RIGHT_SHOULDER, :3]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        # Calculate shoulder width for scaling
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)
        valid = np.isfinite(shoulder_width) & (shoulder_width > 0)

        scale = (
            float(shoulder_width[valid].mean())
            if np.any(valid)
            else self.min_shoulder_scale
        )
        scale = max(scale, self.min_shoulder_scale)

        # Center and scale
        centered = pose_sequence - shoulder_center[:, None, :]
        normalized = centered / scale

        return normalized

    def normalize_with_reference(
        self,
        pose_sequence: np.ndarray,
        reference_center: np.ndarray,
        reference_scale: float,
    ) -> np.ndarray:
        """
        Normalize using provided reference values (for test-time consistency).
        """
        centered = pose_sequence - reference_center[None, None, :]
        normalized = centered / reference_scale
        return normalized


class HandNormalizer:
    """Normalize hand landmarks using wrist-based reference."""

    def __init__(self, min_scale: float = 0.1):
        self.min_scale = min_scale

    def normalize(self, hand_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize hand sequence using wrist-based reference.

        Args:
            hand_sequence: Array of shape (num_frames, 21, 3)

        Returns:
            Normalized hand sequence
        """
        wrist = hand_sequence[:, LandmarkIndices.WRIST, :3]

        # Calculate scale based on hand spread
        palm_points = hand_sequence[:, [0, 5, 9, 13, 17], :3]
        palm_center = palm_points.mean(axis=1)
        palm_spread = np.linalg.norm(
            palm_points - palm_center[:, None, :], axis=-1
        ).mean(axis=1)

        valid = np.isfinite(palm_spread) & (palm_spread > self.min_scale)
        scale = float(palm_spread[valid].mean()) if np.any(valid) else 1.0
        scale = max(scale, self.min_scale)

        # Center at wrist and scale
        centered = hand_sequence - wrist[:, None, :]
        normalized = centered / scale

        return normalized


class TemporalAligner:
    """Align sequences to target length using various strategies."""

    def __init__(self, target_length: int, min_length: int = 10):
        self.target_length = target_length
        self.min_length = min_length

    def pad_or_crop(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad or crop sequence to target length using centered approach.

        Args:
            sequence: Input sequence of shape (seq_len, ...)

        Returns:
            Aligned sequence of shape (target_length, ...)
        """
        seq_len = sequence.shape[0]

        if seq_len == self.target_length:
            return sequence

        if seq_len > self.target_length:
            # Crop centered
            start = max(0, (seq_len - self.target_length) // 2)
            return sequence[start : start + self.target_length]

        # Pad with zeros
        pad_length = self.target_length - seq_len
        pad_shape = (pad_length,) + sequence.shape[1:]
        padding = np.zeros(pad_shape, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)

    def resample(self, sequence: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Resample sequence to different length.

        Args:
            sequence: Input sequence
            scale_factor: Multiplier for sequence length

        Returns:
            Resampled sequence
        """
        new_length = max(self.min_length, int(len(sequence) * scale_factor))
        new_length = min(new_length, self.target_length)

        # Linear interpolation
        indices = np.linspace(0, len(sequence) - 1, new_length)
        resampled = []
        for idx in indices:
            lower = int(np.floor(idx))
            upper = min(lower + 1, len(sequence) - 1)
            weight = idx - lower
            resampled.append((1 - weight) * sequence[lower] + weight * sequence[upper])

        return np.array(resampled)


class Augmentor:
    """Comprehensive augmentation for sign language data."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.temporal_aligner = TemporalAligner(
            config.max_seq_length, config.min_seq_length
        )

    def augment(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray] = None,
        right_hand: Optional[np.ndarray] = None,
        face: Optional[np.ndarray] = None,
        apply_augmentation: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Apply augmentation to all input streams.

        Args:
            body_pose: Body pose sequence (seq_len, 33, 3)
            left_hand: Left hand sequence (seq_len, 21, 3) or None
            right_hand: Right hand sequence (seq_len, 21, 3) or None
            face: Face sequence (seq_len, 468, 3) or None
            apply_augmentation: Whether to apply augmentation

        Returns:
            Tuple of augmented sequences
        """
        if not apply_augmentation or not self.config.augmentation:
            # Just align without augmentation
            body_pose = self.temporal_aligner.pad_or_crop(body_pose)
            if left_hand is not None:
                left_hand = self.temporal_aligner.pad_or_crop(left_hand)
            if right_hand is not None:
                right_hand = self.temporal_aligner.pad_or_crop(right_hand)
            if face is not None:
                face = self.temporal_aligner.pad_or_crop(face)
            return body_pose, left_hand, right_hand, face

        # Apply temporal scaling
        if random.random() < 0.3:
            scale = random.uniform(*self.config.temporal_scale_range)
            body_pose = self._temporal_scale(body_pose, scale)
            if left_hand is not None:
                left_hand = self._temporal_scale(left_hand, scale)
            if right_hand is not None:
                right_hand = self._temporal_scale(right_hand, scale)
            if face is not None:
                face = self._temporal_scale(face, scale)

        # Apply spatial augmentations
        if random.random() < 0.4:
            body_pose = self._add_gaussian_noise(body_pose)
            if left_hand is not None:
                left_hand = self._add_gaussian_noise(left_hand)
            if right_hand is not None:
                right_hand = self._add_gaussian_noise(right_hand)
            if face is not None:
                face = self._add_gaussian_noise(face)

        if random.random() < 0.3:
            angle = np.radians(
                random.uniform(-self.config.rotation_range, self.config.rotation_range)
            )
            body_pose = self._rotate_2d(body_pose, angle)
            if left_hand is not None:
                left_hand = self._rotate_2d(left_hand, angle)
            if right_hand is not None:
                right_hand = self._rotate_2d(right_hand, angle)
            if face is not None:
                face = self._rotate_2d(face, angle)

        if random.random() < 0.3:
            scale = random.uniform(*self.config.scale_range)
            body_pose = self._scale(body_pose, scale)
            if left_hand is not None:
                left_hand = self._scale(left_hand, scale)
            if right_hand is not None:
                right_hand = self._scale(right_hand, scale)
            if face is not None:
                face = self._scale(face, scale)

        # Align to target length
        body_pose = self.temporal_aligner.pad_or_crop(body_pose)
        if left_hand is not None:
            left_hand = self.temporal_aligner.pad_or_crop(left_hand)
        if right_hand is not None:
            right_hand = self.temporal_aligner.pad_or_crop(right_hand)
        if face is not None:
            face = self.temporal_aligner.pad_or_crop(face)

        return body_pose, left_hand, right_hand, face

    def _temporal_scale(self, sequence: np.ndarray, scale: float) -> np.ndarray:
        """Apply temporal scaling."""
        return self.temporal_aligner.resample(sequence, scale)

    def _add_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to sequence."""
        noise = np.random.normal(0, self.config.noise_std, sequence.shape)
        return sequence + noise.astype(sequence.dtype)

    def _rotate_2d(self, sequence: np.ndarray, angle: float) -> np.ndarray:
        """Apply 2D rotation in XY plane."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = sequence.copy()

        # Rotate x and y coordinates (indices 0 and 1)
        x = sequence[:, :, 0] - 0.5
        y = sequence[:, :, 1] - 0.5
        rotated[:, :, 0] = cos_a * x - sin_a * y + 0.5
        rotated[:, :, 1] = sin_a * x + cos_a * y + 0.5

        return rotated

    def _scale(self, sequence: np.ndarray, scale: float) -> np.ndarray:
        """Apply uniform scaling."""
        return sequence * scale


class SignLanguageDataset(Dataset):
    """Dataset for sign language recognition."""

    def __init__(
        self,
        sample_paths: List[str],
        word_to_label: Dict[str, int],
        normalized_dir: str,
        config: DataConfig,
        augment: bool = False,
        mode: str = "train",
        use_hands: bool = True,
        use_face: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            sample_paths: List of paths to video files
            word_to_label: Mapping from words to label indices
            normalized_dir: Directory containing preprocessed .npz files
            config: Data configuration
            augment: Whether to apply augmentation
            mode: 'train', 'val', or 'test'
            use_hands: Include hand landmarks
            use_face: Include face landmarks
        """
        self.sample_paths = sample_paths
        self.word_to_label = word_to_label
        self.normalized_dir = Path(normalized_dir)
        self.config = config
        self.augment = augment and mode == "train"
        self.mode = mode
        self.use_hands = use_hands
        self.use_face = use_face

        self.augmentor = Augmentor(config)
        self.pose_normalizer = PoseNormalizer()
        self.hand_normalizer = HandNormalizer()

        # Parse metadata for all samples
        self.metadata_list = [self._parse_metadata(s) for s in sample_paths]
        self.metadata_list = [m for m in self.metadata_list if m is not None]

    def __len__(self) -> int:
        return len(self.metadata_list)

    def _parse_metadata(self, video_path: str) -> Optional[Dict]:
        """Parse metadata from video filename."""
        path = Path(video_path)
        filename = path.stem
        parts = filename.split("__")

        if len(parts) != 5:
            return None

        return {
            "word": parts[0],
            "signer": parts[1],
            "session": parts[2],
            "repetition": parts[3],
            "grammar": parts[4],
            "full_path": video_path,
        }

    def _get_npz_path(self, metadata: Dict) -> Path:
        """Get path to preprocessed .npz file."""
        filename = f"{metadata['word']}__{metadata['signer']}__{metadata['session']}__{metadata['repetition']}__{metadata['grammar']}.npz"
        return self.normalized_dir / filename

    def _load_raw_pose(
        self, metadata: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Load raw pose data from .npz file.

        Returns:
            body_pose: (seq_len, 33, 3)
            left_hand: (seq_len, 21, 3) or None
            right_hand: (seq_len, 21, 3) or None
            raw_length: Original sequence length
        """
        npz_path = self._get_npz_path(metadata)

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing .npz: {npz_path}")

        data = np.load(npz_path)

        # Determine key names
        keys = list(data.keys())

        # Try to find pose sequence
        if "pose_sequence" in keys:
            pose_sequence = data["pose_sequence"]
        else:
            pose_sequence = data[keys[0]]

        # Reshape if needed
        if pose_sequence.ndim == 2:
            # Already flattened: (seq_len, 99) for body only
            body_flat = pose_sequence
            body_pose = body_flat.reshape(-1, 33, 3)
            left_hand = None
            right_hand = None
        elif pose_sequence.ndim == 3:
            # Multi-stream data
            if pose_sequence.shape[1] == 33:
                body_pose = pose_sequence
                left_hand = None
                right_hand = None
            elif pose_sequence.shape[1] == 75:  # body + hands
                body_pose = pose_sequence[:, :33, :]
                left_hand = pose_sequence[:, 33:54, :]
                right_hand = pose_sequence[:, 54:75, :]
            else:
                # Assume body only
                body_pose = pose_sequence
                left_hand = None
                right_hand = None
        else:
            raise ValueError(f"Unexpected pose shape: {pose_sequence.shape}")

        raw_length = pose_sequence.shape[0]

        # Handle face data (not stored in current format)
        face = None

        return body_pose, left_hand, right_hand, face, raw_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        metadata = self.metadata_list[idx]
        label = self.word_to_label[metadata["word"]]

        try:
            body_pose, left_hand, right_hand, face, raw_length = self._load_raw_pose(
                metadata
            )
        except Exception as e:
            print(f"⚠️  Error loading {metadata['full_path']}: {e}")
            # Return zeros on error
            body_pose = np.zeros((self.config.max_seq_length, 33, 3), dtype=np.float32)
            left_hand = (
                np.zeros((self.config.max_seq_length, 21, 3), dtype=np.float32)
                if self.use_hands
                else None
            )
            right_hand = (
                np.zeros((self.config.max_seq_length, 21, 3), dtype=np.float32)
                if self.use_hands
                else None
            )
            face = (
                np.zeros((self.config.max_seq_length, 468, 3), dtype=np.float32)
                if self.use_face
                else None
            )
            raw_length = 0

        # Apply normalization
        body_pose = self.pose_normalizer.normalize(body_pose)
        if left_hand is not None:
            left_hand = self.hand_normalizer.normalize(left_hand)
        if right_hand is not None:
            right_hand = self.hand_normalizer.normalize(right_hand)

        # Apply augmentation
        body_pose, left_hand, right_hand, face = self.augmentor.augment(
            body_pose, left_hand, right_hand, face, self.augment
        )

        # Flatten landmarks to feature vectors
        body_features = body_pose.reshape(body_pose.shape[0], -1)  # (seq_len, 99)

        if left_hand is not None:
            left_hand = left_hand.reshape(left_hand.shape[0], -1)  # (seq_len, 63)
        if right_hand is not None:
            right_hand = right_hand.reshape(right_hand.shape[0], -1)  # (seq_len, 63)
        if face is not None:
            face = face.reshape(face.shape[0], -1)  # (seq_len, 1404)

        # Create attention mask
        seq_length = min(raw_length, self.config.max_seq_length)
        attention_mask = np.zeros(self.config.max_seq_length, dtype=np.float32)
        attention_mask[:seq_length] = 1

        # Convert to tensors
        result = {
            "body_pose": torch.FloatTensor(body_features),
            "label": torch.LongTensor([label]),
            "attention_mask": torch.FloatTensor(attention_mask),
            "seq_length": torch.LongTensor([seq_length]),
            "word": metadata["word"],
            "signer": metadata["signer"],
            "grammar": metadata["grammar"],
        }

        if self.use_hands:
            result["left_hand"] = (
                torch.FloatTensor(left_hand) if left_hand is not None else None
            )
            result["right_hand"] = (
                torch.FloatTensor(right_hand) if right_hand is not None else None
            )

        if self.use_face:
            result["face"] = torch.FloatTensor(face) if face is not None else None

        return result


def create_data_loaders(
    config: DataConfig,
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    word_to_label: Dict[str, int],
    batch_size: int = 16,
    num_workers: int = 2,
    use_hands: bool = True,
    use_face: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Data configuration
        train_samples: List of training sample paths
        val_samples: List of validation sample paths
        test_samples: List of test sample paths
        word_to_label: Word to label mapping
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_hands: Include hand landmarks
        use_face: Include face landmarks

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SignLanguageDataset(
        sample_paths=train_samples,
        word_to_label=word_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=True,
        mode="train",
        use_hands=use_hands,
        use_face=use_face,
    )

    val_dataset = SignLanguageDataset(
        sample_paths=val_samples,
        word_to_label=word_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=False,
        mode="val",
        use_hands=use_hands,
        use_face=use_face,
    )

    test_dataset = SignLanguageDataset(
        sample_paths=test_samples,
        word_to_label=word_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=False,
        mode="test",
        use_hands=use_hands,
        use_face=use_face,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data pipeline
    config = DataConfig()
    print("✅ DataConfig created")
    print(
        f"   Body dim: {config.body_dim}, Hand dim: {config.hand_dim}, Face dim: {config.face_dim}"
    )

    # Test normalizer
    normalizer = PoseNormalizer()
    dummy_pose = np.random.randn(100, 33, 3).astype(np.float32)
    normalized = normalizer.normalize(dummy_pose)
    print(f"✅ PoseNormalizer test: {dummy_pose.shape} -> {normalized.shape}")

    # Test augmentor
    augmentor = Augmentor(config)
    augmented = augmentor.augment(dummy_pose, apply_augmentation=False)
    print(f"✅ Augmentor test: {augmented[0].shape}")

    print("\n✅ All data pipeline components working correctly")
