import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Konfigurasi DagsHub URI (Silakan ubah dengan URI DagsHub Anda jika menggunakan DagsHub)
# import dagshub
# dagshub.init(repo_owner='Yusuf-Saputrah', repo_name='Sistem-ML-Project', mlflow=True)

def train_tuning():
    # Pastikan autolog dimatikan karena kriteria Skilled/Advance mengharuskan manual logging
    mlflow.sklearn.autolog(disable=True)
    
    # Set eksperimen
    mlflow.set_experiment("Advance_Tuning_RandomForest")
    
    # Load dataset
    df = pd. pd.read_csv('dataset_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Model Dasar
    rf = RandomForestClassifier(random_state=42)
    
    # Grid Search (Hyperparameter Tuning)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        # Training
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Prediksi
        preds = best_model.predict(X_test)
        
        # Metriks
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        # Manual Logging Parameter
        mlflow.log_params(grid_search.best_params_)
        
        # Manual Logging Metrik
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # Artifact 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        # Artifact 2: Feature Importance Plot
        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        sns.barplot(x=importances, y=X.columns)
        plt.title('Feature Importances')
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        
        # Menyimpan model
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        print(f"Model trained. Best Params: {grid_search.best_params_}, Accuracy: {acc}")

if __name__ == "__main__":
    train_tuning()
