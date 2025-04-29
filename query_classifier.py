import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from collections import Counter
import joblib

def load_data(file_path: str) -> pd.DataFrame:
    """Read the JSON dataset into a DataFrame."""
    return pd.read_json(file_path)


class EmbedData:
    """Attach a 384‑d SBERT embedding to each question."""

    def __init__(self, data_path: str, model_name: str = "multi-qa-MiniLM-L6-dot-v1"):
        self.model = SentenceTransformer(model_name)
        self.df = load_data(data_path)

        # basic cleaning
        self.df.drop_duplicates(subset="Question", inplace=True)
        self.df.dropna(subset=["Question", "Category"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def embed(self) -> pd.DataFrame:
        self.df["embedding"] = self.model.encode(
            self.df["Question"].tolist(), show_progress_bar=True
        ).tolist()
        return self.df

class CategoryClassifier:
    """Logistic‑Regression on frozen SBERT vectors + k‑fold CV report."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.X = np.vstack(self.df["embedding"].to_numpy())
        self.y = self.df["Category"].values

        # simple multinomial logistic regression (multi_class parameter deprecated)
        self.clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        )

    def cross_validate(self, desired_folds: int = 5):
        # ensure we don't ask for more folds than the smallest class size
        min_class_count = min(Counter(self.y).values())
        n_splits = max(2, min(desired_folds, min_class_count))
        if n_splits < desired_folds:
            print(
                f"[WARN] Smallest class has only {min_class_count} examples → "
                f"reducing folds to {n_splits}."
            )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        }

        results = cross_validate(
            self.clf,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )
        print(f"\n=== {n_splits}‑fold cross‑validation ===")
        for key in scoring.keys():
            scores = results[f"test_{key}"]
            print(f"{key:15}: {scores.mean():.3f} ± {scores.std():.3f}")

    def fit_full(self):
        self.clf.fit(self.X, self.y)

    def predict(self, question: str, embedder: SentenceTransformer) -> str:
        vec = embedder.encode([question])
        return self.clf.predict(vec)[0]

    def save(self, path: str = "category_classifier.joblib"):
        joblib.dump(self.clf, path)
        print(f"Saved trained classifier to {path}")

def main():
    embedder = EmbedData("data/generated_dataset.json")
    df = embedder.embed()

    classifier = CategoryClassifier(df)
    classifier.cross_validate(desired_folds=5)
    classifier.fit_full()

if __name__ == "__main__":
    main()
