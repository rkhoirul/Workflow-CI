import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


mlflow.sklearn.autolog()

# Memuat dataset yang sudah dibersihkan
from pathlib import Path
print("Memuat dataset...")
# Resolve dataset path by checking likely locations (project-level and repo root)
script_dir = Path(__file__).resolve().parent
candidates = [
    (script_dir / 'dataset_preprocessing' / 'heart_cleaned.csv').resolve(),
    (script_dir.parent / 'dataset_preprocessing' / 'heart_cleaned.csv').resolve(),
]

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset berhasil dibagi menjadi data latih dan uji.")

with mlflow.start_run():
    print("Memulai pelatihan model Logistic Regression...")
    
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")
    
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {accuracy:.4f}")

print("\nEksperimen selesai. Jalankan 'mlflow ui' untuk melihat hasilnya.")
