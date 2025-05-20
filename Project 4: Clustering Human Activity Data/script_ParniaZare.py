import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

file_path = "MHEALTHDATASET/mHealth_subject1.log" 
column_names = [
    "acceleration_chest_x", "acceleration_chest_y", "acceleration_chest_z",
    "electrocardiogram_1", "electrocardiogram_2",
    "acceleration_ankle_x", "acceleration_ankle_y", "acceleration_ankle_z",
    "gyroscope_ankle_x", "gyroscope_ankle_y", "gyroscope_ankle_z",
    "magnetometer_ankle_x", "magnetometer_ankle_y", "magnetometer_ankle_z",
    "acceleration_arm_x", "acceleration_arm_y", "acceleration_arm_z",
    "gyroscope_arm_x", "gyroscope_arm_y", "gyroscope_arm_z",
    "magnetometer_arm_x", "magnetometer_arm_y", "magnetometer_arm_z"
]

try:
    data = pd.read_csv(file_path, sep="\\s+", header=None, names=column_names, nrows=1000)
    print("Dataset Loaded Successfully!")
except FileNotFoundError:
    print("File not found. Check your file path!")

# Pre-process Data
data = data.dropna()  

# Looking for outliers
z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
outliers = (z_scores > 3).any(axis=1)
data_cleaned = data[~outliers]
print(f"Number of outliers detected: {outliers.sum()}")

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# turning to 2D
pca = PCA(n_components=2) 
data_pca = pca.fit_transform(data_scaled)

# Clustering
kmeans = KMeans(n_clusters=5, max_iter=300)  
clusters = kmeans.fit_predict(data_pca)

# Visualize the clustering results
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cluster Visualization with PCA-reduced Data')
plt.colorbar()
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(data_pca, clusters)
print("Silhouette Score: ", silhouette_avg)
