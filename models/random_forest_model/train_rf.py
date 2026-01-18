import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_rf(csv_path='../../data/extracted_features.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    X = df.select_dtypes(include=['number']).drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }

    print("Starting Random Forest Hyperparameter Optimization (Grid Search)")
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    joblib.dump(best_model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Classification Report (Best RF):")
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_rf()