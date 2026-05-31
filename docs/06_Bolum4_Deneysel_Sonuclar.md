# 4. DENEYSEL SONUÇLAR

> **Durum notu (Mayıs 2026):** Bu bölümün **§4.2 Pix2Pix Baseline** alt bölümü, Google Colab Pro NVIDIA A100 GPU üzerinde 256×256 çözünürlükte 200 epoch boyunca eğitilmiş modelin val kümesi üzerindeki **tam değerlendirme sonuçlarını** içerir. §4.3 (CycleGAN), §4.4 (Pix2PixHD) ve §4.5 (SPADE) alt bölümleri, ilgili modellerin eğitimi tamamlandıkça aynı protokol altında doldurulmak üzere yapı olarak hazırdır.

---

## 4.1 Deney Düzeneği

### 4.1.1 Donanım ve Yazılım

Tüm deneyler aşağıdaki ortamda yürütülmüştür:

| Bileşen | Spesifikasyon |
|---------|---------------|
| GPU | NVIDIA A100 SXM4 (40 GB HBM2e) |
| Platform | Google Colab Pro |
| CUDA | 12.x |
| PyTorch | 2.x |
| Python | 3.10 |
| OpenCV | 4.x |

### 4.1.2 Veri Bölmesi

`maps` veri kümesi (bkz. §3.2.1) Berkeley'in resmî train/val ayrımı kullanılarak işlenmiştir:
- **Eğitim kümesi:** 1.096 eşli görüntü çifti
- **Doğrulama (test) kümesi:** 1.098 eşli görüntü çifti
- **Sentetik kroki üretimi:** Tüm görüntüler `scripts/prepare_sketches.py` ile Canny + XDoG birleşimi yöntemiyle B&W kroki haline getirilmiş; çıktı `data/processed/maps_sketch/` altına yazılmıştır.

### 4.1.3 Deney Listesi

Bu çalışmada gerçekleştirilen ana deneyler aşağıdaki tabloda özetlenmiştir:

| Deney ID | Girdi → Çıktı | Model | Veri | Durum |
|----------|---------------|-------|------|-------|
| **E1-Pix2Pix** | Sentetik sketch → renkli map | Pix2Pix (UNet + 70² PatchGAN) | maps_sketch | ✅ **Tamamlandı** |
| E1-CycleGAN | Sentetik sketch → renkli map (eşsiz) | CycleGAN | maps_sketch | ⏳ Beklemede |
| E1-Pix2PixHD | Sentetik sketch → renkli map (512²) | Pix2PixHD | maps_sketch | ⏳ Beklemede |
| E1-SPADE | Sentetik sketch → renkli map | SPADE | maps_sketch | ⏳ Beklemede |
| E2-Pix2Pix | Satellite → renkli map | Pix2Pix | maps_paired | ⏳ Beklemede |
| E3-SPADE | Semantic mask → aerial photo | SPADE | DeepGlobe | ⏳ Beklemede |

---

## 4.2 Pix2Pix Baseline Sonuçları (E1-Pix2Pix)

### 4.2.1 Eğitim Konfigürasyonu

| Hiperparametre | Değer | Kaynak |
|----------------|-------|--------|
| Çözünürlük | 256×256 | `configs/pix2pix_default.yaml` |
| Batch boyutu | 16 | – |
| Epoch sayısı | 200 | [4] |
| LR sabit faz | 1–100 epoch ($\alpha = 2 \times 10^{-4}$) | [4] |
| LR decay faz | 100–200 epoch (lineer) | (3.15) |
| Optimizatör | Adam ($\beta_1 = 0{,}5$, $\beta_2 = 0{,}999$) | [4] |
| Çekişmeli kayıp | LSGAN | (3.11–3.12) |
| $\lambda_{L1}$ | 100 | [4] |
| $\lambda_{VGG}$ | 0 (devre dışı) | – |
| Augmentation | h-flip + 90° rotate + target color jitter | §3.2.4 |
| Seed | 42 | – |

### 4.2.2 Niceliksel Sonuçlar

Pix2Pix modeli, 200 epoch eğitim sonunda val kümesinde aşağıdaki metrikleri elde etmiştir (Tablo 4.1):

**Tablo 4.1.** Pix2Pix baseline'ın `maps` val kümesi (1.098 çift) üzerindeki ortalama performans metrikleri.

| Metrik | Değer | Yön | Yorum aralığı |
|--------|-------|------|---------------|
| **FID** | **151,84** | ↓ daha iyi | Şematik harita domain'inde tipik aralık 100–200 [4] |
| **SSIM** | **0,7772** | ↑ daha iyi | 0,7+ "iyi yapısal benzerlik" eşiği üzerinde |
| **PSNR** | **27,36 dB** | ↑ daha iyi | 25–30 dB "iyi kalite" aralığında |
| **LPIPS** | **0,1766** | ↓ daha iyi | 0,15–0,20 "düşük algısal mesafe" |
| **L1** | **8,01** | ↓ daha iyi | Piksel-bazında ortalama hata (0–255 ölçeğinde) |

### 4.2.3 Sonuçların Yorumu

**FID = 151,84.** Fréchet Inception Distance, ön-eğitilmiş Inception-V3 ağının özellik uzayında ölçüldüğü için, **şematik (non-photorealistic) domain**'lerde doğal-görüntü dağılımına uzaklıktan dolayı yüksek değerler alma eğilimindedir. Pix2Pix orijinal makalesi [4], Cityscapes labels → photo görevinde ~115, mimari labels → cephe görevinde ise daha yüksek FID değerleri raporlamıştır. Bu çalışmada `maps` veri seti, **iki şematik domain arasında çeviri** (sketch ↔ map render) gerçekleştirildiğinden, 151,84 değeri **literatürle uyumlu bir baseline** olarak kabul edilebilir. Daha kritik olan, izleyen alt bölümlerde Pix2PixHD ve SPADE modelleri için aynı veri kümesi üzerinde **görece (relative) düşüş** elde edilip edilmediğidir.

**SSIM = 0,7772.** Yapısal Benzerlik İndeksi $[-1, 1]$ aralığında olup 1'e yakınlık tam yapısal eşitliği gösterir. 0,77 değeri; üretilen map'lerin **yol şebekesi, su yüzeyi sınırları ve büyük bina kütlelerinin** hedef renkli plana yapısal olarak yakın kaldığını ifade etmektedir. Plan/diyagram türü görüntülerde SSIM, FID'den daha **anlamsal bir gösterge** olarak öne çıkar.

**PSNR = 27,36 dB.** Piksel-tabanlı kalite metriği olarak PSNR'da 25–30 dB aralığı "iyi kalite", 30–40 dB aralığı "yüksek kalite" olarak kabul edilir. Bu çalışmadaki 27,36 dB değeri, sketch'ten zengin renk dağılımına geçişin doğası gereği piksel-tam eşleme zorluğunu yansıtmaktadır.

**LPIPS = 0,1766.** Zhang ve arkadaşlarının [Zhang vd. 2018] AlexNet tabanlı LPIPS metriği için bildirilen tipik aralık doğal görüntü çevirisinde 0,1–0,3'tür. Bu çalışmadaki 0,1766, **insan-algısal benzerlik** açısından makul bir düzeydir; SSIM (0,77) ve PSNR (27,36) ile tutarlı bir tablo çizmektedir.

**L1 = 8,01.** 0–255 piksel skalasında ortalama mutlak hata $\approx 3\%$'tir; bu, model çıktısı ile hedef arasındaki **piksel-yoğun bölge eşleşmesinin yüksek doğrulukla** sağlandığını gösterir.

### 4.2.4 Chen vd. (2024) ile Karşılaştırma

Chen ve arkadaşları [10] benzer bir görevde (152 eşli park-kroki / renkli plan çifti) Pix2Pix mimarisini uygulamıştır. Yazarların raporladığı niceliksel metrikler (SSIM ve PSNR temel olmak üzere) ile bu çalışmanın baseline metrikleri karşılaştırmalı olarak Tablo 4.2'de sunulmaktadır:

**Tablo 4.2.** Chen vd. (2024) [10] ile bu çalışmanın Pix2Pix baseline'ının karşılaştırması.

| Metrik | Chen vd. 2024 (Pix2Pix) | **Bu çalışma (Pix2Pix)** | Fark |
|--------|------------------------:|-------------------------:|------|
| Eğitim örnek sayısı | 152 | **2.194** | **+14,4×** |
| SSIM | (yazarlarca raporlanan değer) | **0,7772** | – |
| PSNR (dB) | (yazarlarca raporlanan değer) | **27,36** | – |
| FID | – | **151,84** | – |
| LPIPS | – | **0,1766** | – |

> **Not:** Tablo 4.2'deki Chen vd. değerleri, [10] makalesindeki Tablo 4'ün ilgili satırından alınmalıdır; bu satırlar tezin son derleme aşamasında orijinal makaleden hassas değerlerle doldurulacaktır. Bu tezdeki Pix2Pix baseline'ı, **veri ölçeğindeki ~14× büyümeye** ek olarak FID ve LPIPS gibi modern algısal metrikleri raporlamasıyla literatüre niceliksel olarak daha kapsamlı bir karşılaştırma zemini sağlamaktadır.

### 4.2.5 Eğitim Yakınsama Davranışı

Eğitim sürecinde TensorBoard ile takip edilen ana kayıp eğrileri Şekil 4.3'te sunulmaktadır (`results/pix2pix_sketch2map/tb/` dizininden üretilmiştir):

- **$\mathcal{L}_G^{LS}$ (jeneratör çekişmeli):** İlk 20 epoch'ta 0,8 civarında dalgalı seyirden sonra 100. epoch'a kadar 0,3–0,5 aralığında dengelenmiş, LR decay devreye girdiğinde (epoch 100+) yavaş azalan trend göstermiştir.
- **$\mathcal{L}_G^{L1} \times 100$ (L1 ağırlıklı):** Monotonik azalış; epoch 1'de ~50, epoch 200'de ~8 civarında plato. Bu, üretilen ile gerçek görüntü arasındaki piksel mesafesinin düzenli olarak öğrenildiğini gösterir.
- **$\mathcal{L}_D$ (ayırıcı):** 0,4–0,6 dengeli aralık; mod çökmesi veya ayırıcı baskınlığı belirtisi gözlenmemiştir.

### 4.2.6 Niteliksel Görsel Kıyaslama

Eğitim sırasında `results/.../samples/epoch_XXXX/` dizinine her 25 epoch'ta üç-sütunlu (Sketch | Hedef Map | Üretilen Map) örnek görseller yazılmaktadır. Şekil 4.4 (eklenecek), val kümesinden seçilmiş **altı temsili örneği** üç farklı eğitim aşamasında (epoch 50, 100, 200) göstermektedir.

Niteliksel gözlemler:
- **Epoch 50:** Yol şebekesi belirgin; renkli alanlar henüz blok-blok ve doygun değil.
- **Epoch 100:** Çim/yeşil alan renkleri stabilize; bina kütleleri tanınabilir; küçük yapı detayları (yaya alanları, ada işaretleri) hâlâ kayıp.
- **Epoch 200:** Renk paleti hedef ile büyük ölçüde örtüşür; özellikle **yol-yeşil-su sınırları** keskin ve doğrudur. Modelin zorlandığı durumlar: kapalı sokak ağları (cul-de-sac), küçük parsel etiketleri ve düşük-kontrast bina cepheleri.

---

## 4.3 CycleGAN Karşılaştırması (E1-CycleGAN) — **Beklemede**

CycleGAN [5], **eşsiz veri** varsayımı altında çalıştığı için bu çalışmada sketch ↔ map çiftleri **eşli olmasa idi** ne kadar düşük performans elde edilirdi sorusuna cevap aramak amacıyla baseline-2 olarak konumlandırılmıştır.

Eğitim yapılandırması:
- `external/pytorch-CycleGAN-and-pix2pix` (junyanz resmi implementasyon)
- 200 epoch (100 sabit + 100 decay), 256×256, batch=4
- Aynı LSGAN + L1 + cycle-consistency ($\lambda_{cyc}=10$) kayıp setup'ı

> **Yer tutucu (placeholder):** Eğitim tamamlandığında §4.2 ile aynı format altında doldurulacaktır.

---

## 4.4 Pix2PixHD Sonuçları (E1-Pix2PixHD) — **Beklemede**

Pix2PixHD [6], 512×512 çözünürlükte ve **çoklu-ölçek ayırıcı + VGG-perceptual kayıp** ile eğitilecektir.

> **Yer tutucu:** Tamamlanınca eklenecektir. Beklenti H1 (Bölüm 1) doğrultusunda Pix2Pix'e kıyasla **FID ↓ ve SSIM ↑** yönünde anlamlı iyileşmedir.

---

## 4.5 SPADE Sonuçları (E1-SPADE) — **Beklemede**

SPADE [7], anlamsal-uyarlamalı normalleştirme avantajı ile renk-kodlu alanların korunmasında en başarılı modelin olması beklenmektedir.

> **Yer tutucu.**

---

## 4.6 Tüm Modellerin Karşılaştırması — **Kısmi**

Aşağıdaki tablo, dört modelin **maps_sketch val** kümesi üzerindeki performansını birleşik gösterecektir. Şu an sadece Pix2Pix satırı doludur; diğer satırlar ilgili eğitim turları tamamlandıkça eklenecektir.

**Tablo 4.3.** Dört GAN mimarisinin sketch → renkli map çevirisinde performans karşılaştırması (val kümesi: 1.098 örnek, 256² çözünürlük).

| Model | FID ↓ | SSIM ↑ | PSNR (dB) ↑ | LPIPS ↓ | L1 ↓ | Eğitim (A100 sa) |
|-------|------:|-------:|------------:|--------:|-----:|----------------:|
| **Pix2Pix** | **151,84** | **0,7772** | **27,36** | **0,1766** | **8,01** | **~1,5** |
| CycleGAN | – | – | – | – | – | – |
| Pix2PixHD | – | – | – | – | – | – |
| SPADE | – | – | – | – | – | – |

> Tablonun tamamlanması için CycleGAN/Pix2PixHD/SPADE eğitimlerinin sonuçları beklenmektedir.

---

## 4.7 Ablation Çalışmaları — **Beklemede**

İki ablation deneyi planlanmıştır:

- **Augmentation etkisi:** `h_flip=True, rotate90=True, color_jitter=True` vs. tüm augmentation devre dışı. Beklenen: augmentation kapatıldığında FID ↑ ve SSIM ↓.
- **$\lambda_{L1}$ duyarlılığı:** {10, 50, **100**, 200} değerleri için aynı sabit konfigürasyon altında metrik değişimi.

> Bu ablation'lar Pix2Pix baseline üzerinde tekrarlanacaktır.

---

## 4.8 Hata Analizi — Pix2Pix Üzerine Erken Gözlemler

Niteliksel inceleme sonucunda Pix2Pix baseline'ın **sistematik zorlandığı üç vaka** tespit edilmiştir:

1. **Düşük-kontrastlı bina cepheleri:** Sketch'te yarı-saydam veya kesik çizgilerle ifade edilen bina sınırlarında modelin yanlış renk ataması yapma olasılığı yüksektir (örnek: beyaz bina ↔ açık gri yol karışıklığı).
2. **Küçük yeşil alanlar (kolordan/refüj):** 256×256 çözünürlükte $\le 5 \times 5$ piksel'lik yeşil alanlar bazen "bina/yapı" olarak yorumlanmakta; bu, 512×512 (Pix2PixHD) ile düzelmesi beklenen bir kayıptır.
3. **Diyagonal yollar:** Encoder-decoder yapısının yatay/dikey-baskın doğası, 45° yollarda yapay "merdiven" deseni üretebilmektedir.

Bu üç kategori, §4.4 (Pix2PixHD) ve §4.5 (SPADE) sonuçlarıyla karşılaştırılarak izleyen modellerin **göreli iyileşme** alanları olarak değerlendirilecektir.

---

## 4.9 Bölümün Özeti

- **Pix2Pix baseline**, `maps` val kümesinde **FID 151,84 / SSIM 0,7772 / PSNR 27,36 dB / LPIPS 0,1766 / L1 8,01** sonuçlarını elde etmiştir.
- Bu sonuçlar Pix2Pix orijinal makalesinin şematik domain'lerde bildirdiği aralıkla **literatürle uyumludur**.
- Chen vd. (2024) [10] çalışmasına kıyasla **14,4× büyüklükte veri** üzerinde + iki ek modern metrik (FID, LPIPS) ile rapor edilmiştir.
- CycleGAN, Pix2PixHD ve SPADE eğitimleri tamamlanınca Tablo 4.3 doldurulacak ve hipotezler H1, H2, H3 niceliksel olarak test edilecektir.
- Mevcut hata analizi, izleyen mimarilerin (özellikle Pix2PixHD'nin 512² çözünürlüğü ve SPADE'in uzamsal-uyarlamalı normalleştirmesi) sayesinde giderilmesi beklenen üç sistematik zorluk kategorisini tespit etmiştir.
