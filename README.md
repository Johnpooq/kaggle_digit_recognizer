# Kaggle Digit Recognizer

My solution to the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition — classify handwritten digits (0–9) from 28×28 grayscale images.

## Approach

- **Model:** Support Vector Classifier (SVC), `C=10`, RBF kernel (`gamma="scale"`)
- **Preprocessing:** none — raw pixel values used as-is
- **Validation:** 80/20 train–validation split, stratified, `random_state=3`
- **Final training:** refit on the full training set before predicting on the test set

## Results

- Validation accuracy: ~0.98
- Kaggle public LB score: see [my submissions](https://www.kaggle.com/competitions/digit-recognizer/leaderboard)

## Setup

```bash
pip install -r requirements.txt
