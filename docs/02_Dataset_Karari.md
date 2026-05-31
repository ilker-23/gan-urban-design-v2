# Dataset Kararı (Revize)

**Tarih:** 31 Mayıs 2026
**Karar:** Yeni dataset oluşturulmayacak. Tüm deneyler **hazır + halka açık + tek komutla indirilebilir** datasetler üzerinde yürütülecek.

## Gerekçe
- Süre kısıtı: 4-5 gün; özgün veri toplama (OSM + Mapbox) en az 1.5-2 gün alır.
- Türkiye'ye özgü açık şehir verisinde sürtünme yüksek (kayıt, lisans, kapsama).
- Akademik açıdan **literatürde tanınan dataset** kullanmak hakem itirazını azaltır.

## Seçilen Datasetler

### 1) Birincil: `maps` (Pix2Pix / Berkeley official)
- **Kaynak:** `http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz` (no auth).
- **İçerik:** 1096 train + 1098 val = **2194 eşli çift**; 600×600 px (yan yana 1200×600 olarak depolanır).
- **Çiftler:** Satellite aerial ↔ Google Maps render.
- **Tezimize uyum:** Map tarafı **renk-kodlu plan görünümünde** (yeşil=park, gri=yol, mavi=su, beyaz=bina/yapı). Bu, tez önerisinde Şekil 3'teki "renklendirilmiş plan" tipiyle birebir aynı estetik.
- **Literatür konumu:** Pix2Pix (Isola vd., 2017, CVPR) orijinal makalesinde sunulan resmi dataset → standart referans.

### 2) İkincil: `DeepGlobe Land Cover Classification` (Kaggle)
- **Kaynak:** `kaggle datasets download -d balraj98/deepglobe-land-cover-classification-dataset` (Kaggle hesabı yeterli).
- **İçerik:** 1,146 görsel × 2448×2448 px; 7 sınıflı semantic segmentation mask.
- **Sınıflar:** urban, agriculture, rangeland, **forest**, water, barren, unknown.
- **Tezimize uyum:** "Semantic mask → aerial photo" çevirisi için SPADE/GauGAN deneyini destekler.

## Deneyler Bu İki Dataset ile Tam Karşılığını Bulur

| Deney | Girdi → Çıktı | Dataset | Model | Tezin hangi iddiasını destekler |
|-------|---------------|---------|-------|---------------------------------|
| **E1** | Sentetik kroki → Renkli map | maps | Pix2Pix, CycleGAN | T3 — sketch renklendirme (ana iddia) |
| **E2** | Aerial → Renkli map | maps | Pix2Pix, Pix2PixHD | T1+T2 — plan üretimi + render |
| **E3** | Semantic mask → Aerial foto | DeepGlobe | SPADE/GauGAN | T2 — foto-gerçekçi render |

## Sentetik Kroki Üretimi (E1 için)
Map görüntüsünün **Canny edge detector** veya **XDoG (eXtended Difference of Gaussians)** filtresinden geçirilerek siyah-beyaz çizgi haline çevrilmesi.
- OpenCV'de hazır: `cv2.Canny`, `cv2.ximgproc.createDoGFilter`.
- Bu, sketch-to-image GAN çalışmalarında **standart pratik** (SketchyGAN 2018, Auto-Painter 2018, Chen vd. 2024 dahil).
- Sınırlamamızı tez metninde "veri kısıtı nedeniyle sentetik sketch kullanıldı; gerçek elle çizilmiş 10-20 örnekle test setinde doğrulama yapıldı" olarak açıkça belirtiriz.

## Veri Augmentation
- **Yatay flip** (peyzaj planları için doğal)
- **90°/180°/270° rotasyon**
- **Random crop** (256×256 patch)
- **Color jitter** (sadece renkli plan tarafında, sketch tarafında hayır)
- Hepsi `albumentations` ile tek satırda paired-aware uygulanır.

## Sonuç
Bu iki dataset ile tez 4-5 günde tamamlanabilir:
- Dataset toplama süresi: **~30 dakika** (sadece indirme).
- Önişleme süresi: **~1 saat** (sketch sentezi + train/val/test bölme).
- Eğitim süresi: A100 üzerinde toplam **~12-15 saat** (4 model).
- Yazım süresi: paralel ilerler.
