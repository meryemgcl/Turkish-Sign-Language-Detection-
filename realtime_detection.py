import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import json
import warnings
import os

# Gereksiz TensorFlow loglarını gizlemek için
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

def main():
    # Modeli ve Etiketleri Yükle
    model_path = 'model.h5'
    labels_path = 'labels.json'
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("Hata: model.h5 veya labels.json bulunamadı. Lütfen önce modeli eğitin.")
        return

    print("Model yükleniyor...")
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    # Yeni Mediapipe (Tasks) API modülünü başlat
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # Görsellik için çizim araçları
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands # Sadece bağlantı noktalarının haritası (çizim) için kullanıyoruz

    # Kamerayı aç (0 genellikle dahili web kamerasıdır)
    cap = cv2.VideoCapture(0)
    print("Kamera açılıyor... (Çıkmak için 'q' tuşuna basın)")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break
            
        # Görüntüyü ayna (mirror) efektiyle yatay çevir
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Mediapipe formati
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # El tespiti (Yeni Tasks API)
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks_proto in detection_result.hand_landmarks:
                
                # Koordinatları çıkar
                keypoints = []
                for landmark in hand_landmarks_proto:
                    keypoints.extend([landmark.x, landmark.y])
                
                # Tahmin için veriyi numpy array'e ve doğru boyuta (1, 42) çevir
                keypoints = np.array(keypoints).reshape(1, -1)
                
                # Modelden tahmin al
                prediction = model.predict(keypoints, verbose=0)
                predicted_class_index = str(np.argmax(prediction))
                confidence = np.max(prediction)
                
                # Etiketi JSON dosyasından bul
                predicted_letter = labels_dict.get(predicted_class_index, "Bilinmeyen")
                
                # Ekrana yazdırma ayarları
                text = f"Harf: {predicted_letter} (%{int(confidence*100)})"
                
                # Siyah arkaplan üzerine beyaz yazı
                cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Çizim yapmak istiyorsak (Opsiyonel)
                # Yeni API'den gelen veriyi eski formata çevirerek çizebiliriz ya da doğrudan manuel çizebiliriz.
                
                break 

        # Görüntüyü pencerede göster
        cv2.imshow('Turkce Isaret Dili Tespiti (TID)', frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
