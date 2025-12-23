import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Mengaktifkan autologging MLflow untuk Scikit-learn
mlflow.sklearn.autolog()

# Memuat dataset yang sudah dibersihkan
print("Memuat dataset...")
df = pd.read_csv('../dataset_preprocessing/heart_cleaned.csv')

# Memisahkan fitur (X) dan target (y)
X = df.drop('target', axis=1)
y = df['target']

# Membagi data menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset berhasil dibagi menjadi data latih dan uji.")

# Mulai eksperimen MLflow
with mlflow.start_run():
    print("Memulai pelatihan model Logistic Regression...")
    
    # Model Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Melatih model
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")
    
    # Melakukan prediksi pada data uji
    y_pred = model.predict(X_test)
    
    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {accuracy:.4f}")
    
    # Membuat plot akurasi selama training (contoh visualisasi)
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, accuracy], label="Training Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    
    # Menyimpan plot sebagai gambar .png
    plt.savefig("accuracy_plot.png")
    
    # Menyimpan plot sebagai artefak MLflow
    mlflow.log_artifact("accuracy_plot.png")
    
    print("\nEksperimen selesai. Jalankan 'mlflow ui' untuk melihat hasilnya.")
    # Mencatat akurasi sebagai metrik
    mlflow.log_metric("accuracy", accuracy)