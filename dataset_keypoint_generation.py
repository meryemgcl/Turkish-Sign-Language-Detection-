import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import cv2

def extract_keypoints(dataset_path, output_csv):
    # Mediapipe'ın yeni (Tasks) API'si ile el tespiti başlatıyoruz
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Başlık (header) satırını oluştur: label, x_0, y_0, x_1, y_1 ... x_20, y_20
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}'])
        writer.writerow(header)

        # Veri setindeki her bir klasörü (harfi) döngüye al
        for label in sorted(os.listdir(dataset_path)):
            label_path = os.path.join(dataset_path, label)
            
            if not os.path.isdir(label_path):
                continue
                
            print(f"İşlenen harf: {label}")
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                
                try:
                    # Yeni API resmi doğrudan dosyadan okuyabiliyor
                    image = mp.Image.create_from_file(img_path)
                    detection_result = detector.detect(image)
                    
                    # Eğer elde bir landmark (eklem noktası) tespit edildiyse
                    if detection_result.hand_landmarks:
                        for hand_landmarks in detection_result.hand_landmarks:
                            row = [label]
                            for landmark in hand_landmarks:
                                # x ve y koordinatlarını ekliyoruz
                                row.extend([landmark.x, landmark.y])
                            
                            writer.writerow(row)
                            break  # İlk eli al ve diğerlerine bakma
                except Exception as e:
                    # Bozuk fotoğrafları yoksay
                    pass
                    
    print(f"Tamamlandı! Tüm koordinatlar {output_csv} dosyasına kaydedildi.")

if __name__ == "__main__":
    dataset_dir = "TSL-DATASET/TSL-DATASET" 
    output_file = "keypoint.csv"
    
    if os.path.exists(dataset_dir):
        extract_keypoints(dataset_dir, output_file)
    else:
        print(f"Hata: {dataset_dir} klasörü bulunamadı. Lütfen veri setinin indiğinden emin olun.")
