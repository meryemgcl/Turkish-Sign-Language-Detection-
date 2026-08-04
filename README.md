<h1 align="center">🤟 Turkish Sign Language Detection</h1>

<p align="center">
  <strong>Kameradan alınan anlık görüntülerde elleri tespit eden, eklem (landmark) noktalarını çıkaran ve bu verileri kullanarak gerçek zamanlı Türkçe İşaret Dili (TİD) harf sınıflandırması yapan açık kaynaklı bilgisayarlı görü projesi.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Tasks_API-00A98F?style=flat-square&logo=google&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#nasil-calisir"><strong>Nasıl Çalışır</strong></a> ·
  <a href="#hizli-baslangic">Hızlı Başlangıç</a> ·
  <a href="#ozellikler">Özellikler</a> ·
  <a href="#proje-yapisi">Proje Yapısı</a>
</p>

---

## 📸 Ekran Görüntüsü
> *(Buraya proje çalışırken kameradan alınmış bir ekran görüntüsünü sonradan ekleyebilirsiniz: `![Demo](images/demo.png)`)*

---

## İçindekiler

- [Neden Var](#neden-var)
- [Özellikler](#ozellikler)
- [Nasıl Çalışır](#nasil-calisir)
- [Proje Yapısı](#proje-yapisi)
- [Hızlı Başlangıç](#hizli-baslangic)
- [Veri Seti Teşekkürü](#veri-seti-tesekkuru)

---

## Neden Var

İşitme ve konuşma engelli bireylerle iletişimi kolaylaştırmak ve işaret dilini bilmeyen kişilerin de bu dili anlık olarak anlayabilmesini sağlamak amacıyla geliştirilmiştir. Modern Derin Öğrenme (Deep Learning) tekniklerini kullanarak, standart bir web kamerası aracılığıyla işaret dili harflerini saniyeler içinde yazıya dökebilecek bir altyapı sunar. 

---

## Özellikler

| | |
|---|---|
| 🎯 **Gerçek Zamanlı Tespit** | OpenCV üzerinden saniyede 30+ kare hızıyla anlık harf okuma |
| 🖐️ **Hassas Landmark Çıkarımı** | MediaPipe Tasks API ile elin 21 farklı (x, y) 3D eklem noktasını pürüzsüz tespit etme |
| 🧠 **Derin Öğrenme Modeli** | Kendi eğittiğimiz, hafif fakat yüksek doğruluklu İleri Beslemeli Sinir Ağı (Feedforward NN) |
| 🗃️ **Bağımsız Çalışma** | Sadece CPU üzerinde çalışabilme yeteneği, pahalı GPU'lara ihtiyaç duymaz |
| 🔄 **Kolay Eğitilebilir** | Sadece görselleri klasöre koyarak `dataset_keypoint_generation.py` ile sıfırdan veri seti oluşturulabilir |

---

## Nasıl Çalışır

1. **Koordinat Çıkarımı (Keypoint Generation)** — `dataset_keypoint_generation.py` betiği, veri setindeki (harf klasörlerindeki) görselleri tek tek okur. MediaPipe aracılığıyla her görseldeki elin 21 eklem noktasını bulur. Bu x,y koordinat değerlerini ve ait oldukları harf etiketini (A, B, C...) `keypoint.csv` dosyasına kaydeder.
2. **Modelin Eğitimi (Classifier Training)** — `train_classifier.py` betiği, hazırlanan bu CSV dosyasındaki koordinatları TensorFlow modeline besler. Model, hangi koordinat diziliminin hangi harfe karşılık geldiğini öğrenir ve ağırlıkları `model.h5` olarak kaydeder.
3. **Gerçek Zamanlı Yorumlama (Realtime Detection)** — `realtime_detection.py` kameranızı açar, MediaPipe ile elinizin o anki güncel noktalarını anında tespit eder ve eğitilen modele (model.h5) sorar. Gelen yanıt (harf), ekranda gösterilir.

---

## Proje Yapısı

```
Turkish-Sign-Language-Detection-
│
├── README.md                           # Bu dosya
├── dataset_keypoint_generation.py      # Klasörden koordinat çıkaran script
├── train_classifier.py                 # Yapay zekayı eğiten script
├── realtime_detection.py               # Kameradan anlık tespit yapan script
│
├── hand_landmarker.task                # MediaPipe el tespit model dosyası
├── keypoint.csv                        # (Siz oluşturacaksınız) Veri setinin koordinat tablosu
├── model.h5                            # (Siz oluşturacaksınız) Eğitilmiş Keras modeli
└── labels.json                         # (Siz oluşturacaksınız) Harf indeks etiketleri
```

---

## Hızlı Başlangıç

### 1. Kurulum
Depoyu klonlayın ve klasöre girin:
```bash
git clone https://github.com/meryemgcl/Turkish-Sign-Language-Detection-.git
cd Turkish-Sign-Language-Detection-
```

Gerekli Python kütüphanelerini yükleyin:
```bash
pip install opencv-python mediapipe tensorflow pandas scikit-learn numpy
```

MediaPipe'ın görev (task) dosyasını proje klasörünüze indirin:
[Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

### 2. Kendi Modelinizi Eğitmek İsterseniz
*(Halihazırda eğitilmiş modeliniz yoksa sırasıyla şu adımları izleyin)*

1. İşaret dili harflerinin bulunduğu resim klasörünü (`TSL-DATASET`) ana dizine ekleyin.
2. Verileri CSV'ye çevirin:
   ```bash
   python dataset_keypoint_generation.py
   ```
3. Modeli eğitin:
   ```bash
   python train_classifier.py
   ```

### 3. Gerçek Zamanlı Test (Webcam)
```bash
python realtime_detection.py
```
*Kamerayı kapatmak için klavyeden **q** tuşuna basabilirsiniz.*

---

## Veri Seti Teşekkürü

Bu proje mimarisinin oluşturulmasına zemin hazırlayan veri seti (TSL-DATASET) açık kaynaklı bir araştırmadan elde edilmiştir:
> *Temel, T., & Vural, R. A. (2023). Turkish Sign Language Recognition Using CNN with New Alphabet Dataset.*

---
<p align="center">Made with ❤️ for Accessibility</p>
