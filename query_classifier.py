import pandas as pd
import numpy as np
from sklearn import kmeans
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
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
        
    def embed(self):
        # use SBERT to vectorize the natural language data
        self.data_frame['embeddings'] = self.model.encode(self.data_frame['query'].tolist(), show_progress_bar=True)
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
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(list(self.data_frame['embeddings']))
        
        # create a scatter plot of the clusters
        fig = plt.figure()
        fig.title('3D PCA Visulization of Clusters')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=self.data_frame['cluster'], cmap='viridis', marker='o')
        
        fig.show()

class neural_network_category_classifier():
    #TODO: or maybe alternate to transfer learning model?
    categories = None
    dataframe = None
    
    def __init__(self, categories_path, dataframe):
        # initialize the neural network's layers and load data
        self.categories = load_data(categories_path)
        self.dataframe = dataframe
        
        # define layers
        
    
    def forward(self):
        #TODO define forward pass
        
    def train(self):
        #TODO define training loop

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data
    
def main():
    # Load and embed the data
    embedder_obj = embed_data('data/train.csv')
    data_frame = embedder_obj.embed()
    
    # TODO: loop the clustering process with different numbers of clusters until the best amt of clusters is found (but not overfitted)
    # Perform KMeans clustering
    cluster_data_obj = cluster_data(data_frame, clusters=5)
    
    cluster_data_obj.cluster()
    
    cluster_data_obj.visualize_clusters()
    plt.title('3D PCA of Clusters')
    
    # Save the clustered data to a new CSV file
    # data_frame.to_csv('data/clustered_queries.csv', index=False)

if __name__ == "__main__":
    main()