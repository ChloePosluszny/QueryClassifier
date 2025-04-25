import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class embed_data():
    """
    Vectorize queries using SBERT.
    """
    model = None
    data_frame = None
    
    def __init__(self, data_path, model_name = 'multi-qa-MiniLM-L6-dot-v1'): # default is trained on question and answer pairs
        # initialize the model and load data
        self.model = SentenceTransformer(model_name)
        self.data_frame = load_data(data_path)
        
        # clean data (remove duplicates)
        self.data_frame.drop_duplicates(subset = "Question", keep = "first", inplace = True)

        # remove any rows with NaN values
        self.data_frame.dropna(inplace = True)
        
        self.data_frame.reset_index(drop = True, inplace = True)
        
    def embed(self):
        # use SBERT to vectorize the natural language data
        self.data_frame['embeddings'] = self.model.encode(self.data_frame['Question'], show_progress_bar = True).tolist()
        return self.data_frame
    
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
        self.model.fit(list(self.data_frame['embeddings']))
        
        # assign cluster labels to the data frame
        self.data_frame['cluster'] = self.model.labels_
        
        return self.data_frame
    
    def visualize_clusters(self):
        # visualize the clusters using PCA
        
        # reduce vectors to three dimensions for visualization
        pca = PCA(n_components = 3)
        reduced_data = pca.fit_transform(list(self.data_frame['embeddings']))
        
        # create a scatter plot of the clusters
        fig = plt.figure()
        plt.title(f'3D PCA Visulization of {self.model.n_clusters} Clusters')
        
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c = self.data_frame['cluster'], cmap = 'viridis', marker = 'o')
        
        plt.show()
        
    def print_clusters(self):
        # print the clusters
        for i in range (self.model.n_clusters):
            print(f"Cluster {i}: {self.data_frame[self.data_frame['cluster'] == i]['Question'].values}")

# class neural_network_category_classifier():
#     #TODO: or maybe alternate to transfer learning model? <- yea i think transfer learning is smarter
#     categories = None
#     dataframe = None
    
#     def __init__(self, categories_path, dataframe):
#         # initialize the neural network's layers and load data
#         self.categories = dataframe["Category"]
#         self.dataframe = dataframe
        
#         # define layers
        
    
#     def forward(self):
#         #TODO define forward pass
        
#     def train(self):
#         #TODO define training loop

def load_data(file_path):
    """
    Load the dataset from a JSON file.
    """
    data = pd.read_json(file_path)
    return data
    
def main():
    # Load and embed the data
    embedder_obj = embed_data("data/generated_dataset.json")
    data_frame = embedder_obj.embed()
    
    best = (0, 0)
    for i in range (2, 20):
        # Perform KMeans clustering
        cluster_data_obj = cluster_data(data_frame, clusters = i)
        
        cluster_data_obj.cluster()
        
        # Calculate the silhouette score
        score = silhouette_score(data_frame['embeddings'].tolist(), cluster_data_obj.model.labels_)
        if score > best[1]:
            best = (i, score)
    
    # Perform KMeans clustering
    cluster_data_obj = cluster_data(data_frame, clusters=best[0])
    
    cluster_data_obj.cluster()
    
    print(f"Silhouette Score for {best[0]} clusters: {best[1]}")
    
    cluster_data_obj.visualize_clusters()

if __name__ == "__main__":
    main()