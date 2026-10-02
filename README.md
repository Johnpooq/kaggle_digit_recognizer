# Kaggle Digit Recognizer

My solution to the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition — classify handwritten digits (0–9) from 28×28 grayscale images.

## Approach

- **Model:** Support Vector Classifier (SVC), `C=10`, RBF kernel (`gamma="scale"`)
- **Validation:** 80/20 train/validation split, stratified, `random_state=3`
- **Final training:** refit on the full training set before predicting on the test set

## Results

- Validation accuracy: ~0.98
- Kaggle public LB score: see the [competition leaderboard](https://www.kaggle.com/competitions/digit-recognizer/leaderboard)

## Getting the data

This project uses the Digit Recognizer dataset. **The data is not included in this repo.**

### Option A — download from the Kaggle website

1. Go to https://www.kaggle.com/competitions/digit-recognizer/data
2. Sign in (free Kaggle account)
3. Click **Download All** — saves `digit-recognizer.zip`
4. Unzip it in the project root so you have:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

### Option B — download with the Kaggle CLI

Install and authenticate once:

```bash
pip install kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
# and place it at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
