from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 3

def main():
    train = pd.read_csv(DATA_DIR/"train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = train.drop(columns = "label")
    labels = train["label"]

    training_features, validation_features, training_labels,validation_lables = (
        train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state = RANDOM_STATE,
            stratify= labels,
        )
    )

    # model = RandomForestClassifier(n_estimators=200,random_state=RANDOM_STATE,n_jobs=-1)
    model = SVC(C=10, gamma="scale")
    model.fit(training_features,training_labels)
    validations = model.predict(validation_features)
    validation_score = accuracy_score(validation_lables,validations)
    print(f"Validation score:{validation_score}")

    model.fit(features,labels)
    predicitons = model.predict(test[features.columns])
    sample = pd.read_csv(DATA_DIR/"sample_submission.csv")
    sample["Label"] = predicitons
    output_path = DATA_DIR / "submission.csv"
    sample.to_csv(output_path,index=False)
    print(f"file saved to {output_path}")

if __name__ == "__main__":
    main()
