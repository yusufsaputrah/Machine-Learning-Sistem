import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import itertools

# Konfigurasi DagsHub URI
import dagshub
dagshub.init(repo_owner='yusufsaputrah', repo_name='Machine-Learning-Sistem', mlflow=True)

def train_tuning():
    # Menggunakan manual logging untuk memenuhi kriteria Skilled
    
    # Set eksperimen
    mlflow.set_experiment("Advance_Tuning_RandomForest")
    
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'dataset_preprocessing.csv'))
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter Grid
    n_estimators_list = [50, 100, 150]
    max_depth_list = [None, 10, 20]
    
    best_acc = 0
    best_model = None
    best_params = {}
    best_preds = None

    # Loop over all combinations to create multiple MLflow runs
    for n_estimators, max_depth in itertools.product(n_estimators_list, max_depth_list):
        run_name = f"RF_n{n_estimators}_d{max_depth if max_depth else 'None'}"
        with mlflow.start_run(run_name=run_name):
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(X_train, y_train)
            
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            # Manual Logging Parameter, Metric, dan Model
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(rf, "model")
            
            print(f"Run: {run_name} | Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_model = rf
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_preds = preds

    # --- ARTEFAK TAMBAHAN UNTUK KRITERIA ADVANCED ---
    # Log artefak terbaik di run terpisah atau bisa di run terakhir. 
    # Lebih baik kita buat satu run khusus untuk Best Model beserta artefaknya agar menonjol.
    with mlflow.start_run(run_name="Best_Model_Artifacts"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_acc)
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Artifact Tambahan 1: Test Confusion Matrix Plot
        cm = confusion_matrix(y_test, best_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(base_dir, 'test_confusion_matrix.png')
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Artifact Tambahan 2: Feature Importance Plot
        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        sns.barplot(x=importances, y=X.columns)
        plt.title('Feature Importances')
        fi_path = os.path.join(base_dir, 'feature_importance.png')
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        
        # Artifact Tambahan 3: Classification Report CSV
        from sklearn.metrics import classification_report
        report = classification_report(y_test, best_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        cr_path = os.path.join(base_dir, 'classification_report.csv')
        report_df.to_csv(cr_path)
        mlflow.log_artifact(cr_path)
        
        print(f"\nBest Model logged with Params: {best_params}, Accuracy: {best_acc}")

if __name__ == "__main__":
    train_tuning()
