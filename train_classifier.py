import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_model(csv_path='keypoint.csv', model_save_path='model.h5', labels_save_path='labels.json'):
    print("Veri seti yükleniyor...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Hata: {csv_path} dosyası bulunamadı. Lütfen önce keypoint çıkarma scriptini çalıştırın.")
        return

    # Özellikler (X) ve Etiketler (y) olarak ayır
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    # Etiketleri sayısallaştır (Örn: 'A' -> 0, 'B' -> 1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    num_classes = len(label_encoder.classes_)
    print(f"Toplam tespit edilen sınıf sayısı: {num_classes}")

    # Etiket haritalamasını kaydet (daha sonra tahmin yaparken gerekecek)
    label_mapping = {int(index): str(label) for index, label in enumerate(label_encoder.classes_)}
    with open(labels_save_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    print(f"Etiket eşleşmeleri {labels_save_path} dosyasına kaydedildi.")

    # Veriyi eğitim (%80) ve test (%20) olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Model oluşturuluyor...")
    # İleri Beslemeli Sinir Ağı (Feedforward Neural Network) Mimarisi
    model = Sequential([
        Dense(128, activation='relu', input_shape=(42,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Aşırı öğrenmeyi (overfitting) önlemek için EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # Test seti üzerinde doğruluk oranını ölç
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Eğitimi Tamamlandı! Test Seti Doğruluğu (Accuracy): {accuracy * 100:.2f}%")

    # Modeli kaydet
    model.save(model_save_path)
    print(f"Model başarıyla '{model_save_path}' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()
