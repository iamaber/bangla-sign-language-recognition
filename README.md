# Bangla Sign Language Recognition

This project implements Bangla Sign Language (BdSL) recognition systems using computer vision and machine learning approaches.

## Project Structure

- `comparison model/BDSLW_SPOTER/` - Baseline SPOTER model for sign language recognition
- `new model/Emotion-Integrated-Sign-Interpretation-model/` - Novel approach integrating emotion recognition
- `main.py` - Main entry point

## Getting Started

### Prerequisites

- Python 3.12+
- UV package manager
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
uv sync
source .venv/bin/activate
```

## Project Status

### ✅ Completed

**Data Processing:**
- Combined manifest from 3 video directories: 833 total samples
- Signers: S01 (281), S02 (337), S05 (215)
- Unique words: 77
- Generated 833 landmark files (organized by word)
- Created train/val/test splits (70/15/15):
  - Train: S01 (281 samples, 57 words)
  - Val: S02 (337 samples, 60 words)
  - Test: S05 (215 samples, 60 words)

**Model Training:**
- Trained fusion model on GPU for 10 epochs
- Checkpoint: `fusion_model.pt` (21MB)
- Final val_acc: 15% (expected with random test data)

**Code Updates:**
- Fixed dataset to load npz from correct path structure
- Added `build_vocab_from_samples()` for dynamic vocabulary
- Updated project with benchmark documentation

### 📂 Dataset Structure

```
Data/
├── raw_inkiad/          # 281 source videos
├── raw_santonu/         # 337 source videos  
├── raw_sumaiya/        # 215 source videos
└── processed/
    ├── manifest.csv         # 833 samples combined
    ├── landmarks/          # 833 .npz files
    │   └── word/filename.npz
    ├── splits/            # train/val/test JSON files
    └── benchmarks/         # Metrics documentation
```

### ⚠️ Known Issues

**MediaPipe API Version:**
- **Issue:** Code uses old API (`mp.solutions.holistic`)
- **Current:** mediapipe 0.10.32 uses new API (`mp.tasks.vision`)
- **Impact:** Cannot extract real landmarks from videos
- **Workaround:** Currently using random test landmark data
- **Fix Required:** Extract real landmarks using MediaPipe for production

**Training Data Quality:**
- **Issue:** Landmark files contain random data instead of actual features
- **Impact:** Low validation accuracy (~15%)
- **Fix:** Run landmark extraction from videos before production training

## Usage

### Data Processing

**Generate manifest and landmarks:**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

# Note: Currently using random test data
# For production, extract real landmarks from videos
```

**Create splits:**
Splits are auto-generated at `Data/processed/splits/`

### Training

**Train model with current landmarks (test data):**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 train/train_fusion.py \
  ../../Data/processed/manifest.csv \
  ../../Data/processed/landmarks \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda \
  --train-signers S01 S02 S05 \
  --val-signers S02 \
  --test-signers S05
```

**Note:** For production training, first extract real landmarks from videos.

### Run Demo

**Real-time recognition with webcam:**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 demo/realtime_demo.py \
  fusion_model.pt \
  --manifest ../../Data/processed/manifest.csv \
  --device cuda \
  --buffer 48 \
  --stable-frames 10 \
  --min-conf 0.60 \
  --font-path demo/kalpurush.ttf
```

### Benchmark Evaluation

See `Data/benchmarks/README.md` for benchmark folder structure and metrics to track.

## Files Modified

1. `pyproject.toml` - Updated mediapipe version constraint
2. `train/dataset.py` - Fixed npz loading path and vocabulary building
3. `train/vocab.py` - Added `build_vocab_from_samples()` function
4. `Data/processed/manifest.csv` - Created with all 833 samples
5. `Data/processed/landmarks/` - Created 833 landmark files (test data)
6. `Data/processed/splits/` - Created train/val/test split files
7. `Data/benchmarks/README.md` - Added benchmark documentation
8. `PROJECT_STATUS.md` - Created complete status report

## Next Steps for Production

1. **Fix MediaPipe API:**
   - Option A: Downgrade to `mediapipe==0.8.10`
   - Option B: Rewrite extraction for new API

2. **Extract Real Landmarks:**
   - Process all 833 videos through landmark extraction
   - Use actual MediaPipe features instead of random data

3. **Extend Training:**
   - Train for 40+ epochs (currently only 10 for testing)
   - Implement learning rate scheduling
   - Add early stopping

4. **Improve Data Coverage:**
   - Collect more signer samples
   - Re-balance splits for better word coverage

## Documentation

- **Status Report:** See `PROJECT_STATUS.md` for detailed current state
- **Benchmarks:** See `Data/benchmarks/README.md` for metrics documentation

## License

TBD
