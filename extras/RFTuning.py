import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    'n_estimators': [200, 300, 400, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 50, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'class_weight': ['balanced', None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print("Random Forest Tuning")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")