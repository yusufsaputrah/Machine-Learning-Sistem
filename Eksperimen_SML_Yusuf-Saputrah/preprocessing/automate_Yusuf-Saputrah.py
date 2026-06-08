import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Pisahkan fitur dan target
    X = df[['age', 'income']]
    y = df['target']
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Gabungkan kembali
    df_preprocessed = pd.DataFrame(X_scaled, columns=['age_scaled', 'income_scaled'])
    df_preprocessed['target'] = y.values
    return df_preprocessed

def save_data(df, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Jalur relatif
    raw_path = "../dataset_raw.csv"
    output_path = "dataset_preprocessing.csv"
    
    print("Loading raw data...")
    df_raw = load_data(raw_path)
    
    print("Preprocessing data...")
    df_clean = preprocess_data(df_raw)
    
    save_data(df_clean, output_path)
    print("Preprocessing completed!")
