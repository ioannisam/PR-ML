import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

TRAIN_DATA = os.path.join(os.path.dirname(__file__), '../datasets/datasetTV.csv')
NUM_FEATURES = 224

# load dataset
train = pd.read_csv(TRAIN_DATA, header=None)
feature_columns = [f'feature_{i+1}' for i in range(NUM_FEATURES)]
train.columns = feature_columns + ['label']
X_train = train[feature_columns]
y_train = train['label']

# select features with high absolute correlation with the target
correlation_matrix = X_train.corrwith(y_train)
correlation_threshold = 0.1
selected_features = correlation_matrix[abs(correlation_matrix) > correlation_threshold].index
X_train_reduced = X_train[selected_features]

# standardize features
scaler = StandardScaler()
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)

# define parameter grid to search
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.0001],
    'early_stopping': [True, False],
    'max_iter': [200, 300, 500]
}

# initialize MLPClassifier
model = MLPClassifier(random_state=42)

# perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_reduced_scaled, y_train)

print("Neural Network Tuning")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")