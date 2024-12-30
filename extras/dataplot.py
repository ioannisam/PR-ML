import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

data_path = os.path.join(os.path.dirname(__file__), '../datasets/datasetTV.csv')
image_path = os.path.join(os.path.dirname(__file__), '../assets/datasetTV.png')

TRAIN_DATA = data_path
NUM_FEATURES = 224
SAMPLE_SIZE = 4500

# Load dataset
train = pd.read_csv(TRAIN_DATA, header=None)
feature_columns = [f'feature_{i+1}' for i in range(NUM_FEATURES)]
train.columns = feature_columns + ['label']
X_train = train[feature_columns]
y_train = train['label']

# Take a random subset
subset = train.sample(n=SAMPLE_SIZE, random_state=42)
X_subset = subset[feature_columns]
y_subset = subset['label']

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_subset)

# Visualize the data in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_subset, cmap='viridis', alpha=0.8)
ax.set_title("3D PCA Visualization of Classes")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
colorbar.set_label("Label")

plt.savefig(image_path)
plt.show()