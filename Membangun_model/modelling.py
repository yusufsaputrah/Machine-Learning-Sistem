import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train():
    # Mengaktifkan autologging untuk Scikit-learn
    mlflow.sklearn.autolog()
    
    # Load dataset hasil preprocessing
    df = pd.read_csv('dataset_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Memulai MLflow run
    with mlflow.start_run(run_name="Basic_RandomForest"):
        # Model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training (Autolog akan mencatat parameter, metrik, dan model)
        clf.fit(X_train, y_train)
        
        # Prediksi
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    train()
