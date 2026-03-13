# fix_model.py - Recreate model locally
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


# Load data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create pipeline
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsRegressor())
])

# Grid search
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Save properly
best_model = grid_search.best_estimator_

with open('california_knn_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n Model saved! Best params: {grid_search.best_params_}")
print(f"   CV Score: {grid_search.best_score_:.4f}")

# Verify immediately
with open('california_knn_pipeline.pkl', 'rb') as f:
    test_model = pickle.load(f)
    print(f"   Verification: {type(test_model)} loaded successfully")
