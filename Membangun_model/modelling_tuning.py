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

# Konfigurasi DagsHub URI
import dagshub
dagshub.init(repo_owner='yusufsaputrah', repo_name='Machine-Learning-Sistem', mlflow=True)

def train_tuning():
    # Mengaktifkan autolog untuk memenuhi kriteria Skilled (menghasilkan estimator.html, metric_info.json, dll)
    mlflow.sklearn.autolog()
    
    # Set eksperimen
    mlflow.set_experiment("Advance_Tuning_RandomForest")
    
    # Load dataset
    df = pd.read_csv('dataset_preprocessing.csv')
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
        
        # --- ARTEFAK TAMBAHAN UNTUK KRITERIA ADVANCED ---
        
        # Artifact Tambahan 1: Test Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('test_confusion_matrix.png')
        mlflow.log_artifact('test_confusion_matrix.png')
        
        # Artifact Tambahan 2: Feature Importance Plot
        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        sns.barplot(x=importances, y=X.columns)
        plt.title('Feature Importances')
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        
        # Artifact Tambahan 3: Classification Report CSV
        from sklearn.metrics import classification_report
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv')
        mlflow.log_artifact('classification_report.csv')
        
        print(f"Model trained. Best Params: {grid_search.best_params_}, Accuracy: {acc}")

if __name__ == "__main__":
    train_tuning()
