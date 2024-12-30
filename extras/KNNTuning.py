import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

file_path = os.path.join(os.path.dirname(__file__), '../datasets/datasetTV.csv')
NUM_FEATURES = 224
TRAIN_DATA = file_path

# Load dataset
train = pd.read_csv(TRAIN_DATA, header=None)
feature_columns = [f'feature_{i+1}' for i in range(NUM_FEATURES)]
train.columns = feature_columns + ['label']
X_train = train[feature_columns]
y_train = train['label']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define the parameter grid to search
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski', 'cosine'],
    'knn__p': [1, 2],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize the KNeighborsClassifier
model = KNeighborsClassifier()

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print("K-Nearest Neighbors Tuning")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")