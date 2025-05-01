import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from collections import Counter
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV

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
        self.df["Embedding"] = self.model.encode(
            self.df["Question"].tolist(), show_progress_bar=True
        ).tolist()
        return self.df
    
class cluster_data():
    """
    Cluster the data using a k-means alogrithm.
    """
    model = None
    data_frame = None
    
    def __init__(self, data_frame, clusters=5):
        # initialize the model
        self.model = KMeans(n_clusters = clusters, random_state = 42)
        self.data_frame = data_frame
        
    def cluster(self):
        # fit the model to the data
        self.model.fit(list(self.data_frame['Embedding']))
        
        # assign cluster labels to the data frame
        self.data_frame['Cluster'] = self.model.labels_
        
        return self.data_frame
    
    def visualize_clusters(self):
        # visualize the clusters using PCA
        
        # reduce vectors to three dimensions for visualization
        pca = PCA(n_components = 3)
        reduced_data = pca.fit_transform(list(self.data_frame['Embedding']))
        
        # create a scatter plot of the clusters
        fig = plt.figure()
        plt.title(f'3D PCA Visulization of {self.model.n_clusters} Clusters')
        
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c = self.data_frame['Cluster'], cmap = 'viridis', marker = 'o')
        
        plt.show()
        
    def print_clusters(self):
        # print the clusters
        for i in range (self.model.n_clusters):
            print(f"Cluster {i}: {self.data_frame[self.data_frame['Cluster'] == i]['Question'].values}")

class CategoryClassifier:
    """Logistic‑Regression on frozen SBERT vectors + k‑fold CV report."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        # l2-normalize the cached embeddings right away
        self.X_all = normalize(np.vstack(self.df["Embedding"].to_numpy()))

        # default search grid
        self.param_grid = {
            "C": np.logspace(-3, 3, 13),
            "class_weight": [None, "balanced"]
        }
        self.best_params_ = {"C": 1.0, "class_weight": None}
        self.clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            multi_class="multinomial",
        )

    @staticmethod
    def l2_normalize(arr: np.ndarray) -> np.ndarray:
        return normalize(arr)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, desired_folds: int = 5):
        # ensure we don't ask for more folds than the smallest class size
        min_class_count = min(Counter(y).values())
        n_splits = max(2, min(desired_folds, min_class_count))
        if n_splits < desired_folds:
            print(
                f"[WARN] Smallest class has only {min_class_count} examples → "
                f"reducing folds to {n_splits}."
            )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X_norm = normalize(X)
        base_clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", n_jobs=-1
        )
        gs = GridSearchCV(
            estimator=base_clf,
            param_grid=self.param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        gs.fit(X_norm, y)
        self.best_params_ = gs.best_params_
        print(f"\n[GRID] Best params → {self.best_params_} "
              f"(f1_macro = {gs.best_score_:.3f})")
        self.clf = LogisticRegression(
            **self.best_params_,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            multi_class="multinomial",
        )
        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        }
        results = cross_validate(
            self.clf,
            X_norm,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )
        print(f"\n=== {n_splits}-fold CV (best model) ===")
        for key in scoring.keys():
            scores = results[f"test_{key}"]
            print(f"{key:15}: {scores.mean():.3f} ± {scores.std():.3f}")

    def fit_full(self, X: np.ndarray, y: np.ndarray):
        X_norm = self.l2_normalize(X)
        self.clf.fit(X_norm, y)

    def predict(self, question: str, embedder: SentenceTransformer) -> str:
        vec = self.l2_normalize(embedder.encode([question]))
        return self.clf.predict(vec)[0]
    
    def predict_embedded(self, embedding: np.ndarray) -> str:
        return self.clf.predict(self.l2_normalize(embedding))[0]
    
    def predict_embedded_batch(self, embeddings: np.ndarray) -> list:
        return self.clf.predict(self.l2_normalize(embeddings)).tolist()

    def save(self, path: str = "category_classifier.joblib"):
        joblib.dump(self.clf, path)
        print(f"Saved trained classifier to {path}")

def main():
    embedder = EmbedData("data/generated_dataset.json")
    data_frame = embedder.embed()
    
    best = (0, 0)
    for i in range (2, 20):
        # Find the best number of clusters using silhouette score
        cluster_data_obj = cluster_data(data_frame, clusters = i)
        
        cluster_data_obj.cluster()
        
        # Calculate the silhouette score
        score = silhouette_score(data_frame['Embedding'].tolist(), cluster_data_obj.model.labels_)
        if score > best[1]:
            best = (i, score)
    
    # Perform KMeans clustering
    cluster_data_obj = cluster_data(data_frame, clusters=best[0])
    cluster_data_obj.cluster()
    
    print(f"Silhouette Score for {best[0]} clusters: {best[1]}")   
    cluster_data_obj.visualize_clusters()
    
    centroids_df = pd.DataFrame({'Embedding': [list(center) for center in cluster_data_obj.model.cluster_centers_]})
    centroids_df['Cluster'] = [i for i in range(cluster_data_obj.model.n_clusters)]
    centroids_df['Questions'] = [data_frame[data_frame['Cluster'] == i]['Question'].values for i in range(cluster_data_obj.model.n_clusters)]
    
    print(centroids_df.head())
    

    # Classify the centroids using logisitic regression
    classifier = CategoryClassifier(centroids_df)
    classifier.cross_validate(np.vstack(data_frame['Embedding'].to_numpy()), data_frame['Category'].values, desired_folds=5)
    classifier.fit_full(np.vstack(data_frame['Embedding'].to_numpy()), data_frame['Category'].values)
    
    centroids_df['Category'] = classifier.predict_embedded_batch(np.vstack(centroids_df['Embedding'].to_numpy()))
    
    # Print the clusers and their predicted categories
    for category in centroids_df['Category']:
        cluster = centroids_df[centroids_df['Category'] == category]['Cluster'].values[0]
        questions = centroids_df[centroids_df['Category'] == category]['Questions'].values
        
        print(f"\nCluster: {cluster} \nCategory: {category} \nQuestions: {questions}")
        
    # TODO: workout some sort of accuracy metric

if __name__ == "__main__":
    main()
