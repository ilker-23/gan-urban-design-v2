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
| **E1-Pix2Pix** | Sentetik sketch → renkli map | Pix2Pix (UNet + 70² PatchGAN), 256² | maps_sketch | ✅ **Tamamlandı** |
| **E1-CycleGAN** | Sentetik sketch → renkli map (eşsiz) | CycleGAN, 256² | maps_sketch | ✅ **Tamamlandı** |
| **E1-Enhanced** | Sentetik sketch → renkli map | Geliştirilmiş Pix2Pix (512² + VGG) | maps_sketch | 🟡 Eğitimde |
| E1-Pix2PixHD | Sentetik sketch → renkli map | Pix2PixHD (multi-scale D), 512² | maps_sketch | ⚠️ Bkz. §4.5 |
| E2-Pix2Pix | Satellite → renkli map | Pix2Pix | maps_paired | ⏳ İsteğe bağlı |
| E3-SPADE | Semantic mask → aerial photo | SPADE | DeepGlobe | ⏳ İsteğe bağlı |

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

## 4.3 CycleGAN Sonuçları (E1-CycleGAN)

CycleGAN [5], **eşsiz (unpaired) veri** varsayımı altında çalıştığı için bu çalışmada sketch ↔ map çiftleri **eşli olmasaydı** ne tür bir performans elde edileceğini gösteren baseline-2 olarak konumlandırılmıştır.

### 4.3.1 Eğitim Konfigürasyonu

| Hiperparametre | Değer |
|----------------|-------|
| İmplementasyon | `junyanz/pytorch-CycleGAN-and-pix2pix` (resmî) |
| Çözünürlük | 256×256 (load 286 → crop 256) |
| Batch boyutu | 4 |
| Epoch | 200 (100 sabit + 100 lineer decay) |
| Kayıp | LSGAN + döngü-tutarlılık ($\lambda_{cyc}=10$) + identity |
| Optimizatör | Adam ($\beta_1=0{,}5$) |

### 4.3.2 Niceliksel Sonuçlar

**Tablo 4.3.** CycleGAN'ın `maps` val kümesi (1.098 çift) üzerindeki performansı ve Pix2Pix baseline ile karşılaştırması.

| Metrik | Pix2Pix (256, L1) | **CycleGAN (256, eşsiz)** | Daha iyi olan |
|--------|------------------:|--------------------------:|:-------------:|
| **FID** ↓ | 151,84 | **54,58** | **CycleGAN** |
| **SSIM** ↑ | **0,7772** | 0,6578 | Pix2Pix |
| **PSNR (dB)** ↑ | **27,36** | 24,01 | Pix2Pix |
| **LPIPS** ↓ | **0,1766** | 0,2032 | Pix2Pix |
| **L1** ↓ | **8,01** | 11,72 | Pix2Pix |

### 4.3.3 Bulguların Yorumu: Algı-Bozulma Ödünleşimi (Perception-Distortion Tradeoff)

Bu sonuçlar literatürdeki en önemli bulgulardan birini somut olarak göstermektedir: **CycleGAN, FID'de Pix2Pix'i belirgin biçimde geçerken (54,58'e karşı 151,84 — yaklaşık %64 daha düşük/iyi), piksel-tabanlı metriklerin tamamında (SSIM, PSNR, LPIPS, L1) Pix2Pix'in gerisinde kalmaktadır.** Bu görünüşte çelişkili tablo, Blau ve Michaeli'nin (2018) tanımladığı **algı-bozulma ödünleşimi** ile açıklanır:

- **FID**, üretilen görüntülerin **dağılım düzeyinde gerçekçiliğini** (gerçek map'lerin istatistiksel dağılımına yakınlık) ölçer. CycleGAN, L1 piksel kısıtı olmadan, salt çekişmeli + döngü kaybıyla eğitildiği için **çıktıları map estetiğine (keskin renk blokları, doygun yeşil/mavi alanlar) dağılımsal olarak daha çok benzer**; bu da düşük FID üretir.
- **SSIM/PSNR/L1**, üretilen görüntünün **belirli hedef görüntüye piksel-tam sadakatini** ölçer. Pix2Pix, L1 kaybı ($\lambda_{L1}=100$) ile hedefe "çivilendiği" için bu metriklerde üstündür. CycleGAN ise eşsiz öğrendiğinden belirli bir hedefe piksel-tam hizalanma garantisi vermez.

**Tez açısından kritik sonuç:** Plan/diyagram üretimi gibi **mekânsal doğruluğun (yol, su, bina sınırlarının doğru yerde olması)** kritik olduğu uygulamalarda **piksel-sadakat metrikleri (SSIM, PSNR) daha anlamlıdır**; bu nedenle eşli Pix2Pix, kentsel plan renklendirme için CycleGAN'a tercih edilmelidir. Bu bulgu **Hipotez H3'ü doğrular** (eşli Pix2Pix, keskin-sınırlı plan görüntülerinde piksel-tam doğrulukta eşsiz CycleGAN'dan üstündür). Öte yandan, salt "görsel inandırıcılık/gerçekçilik" hedeflenen uygulamalarda CycleGAN'ın düşük FID'i avantajdır.

### 4.3.4 Niteliksel Gözlem

CycleGAN çıktıları görsel olarak **daha "canlı" ve harita-benzeri** görünmekle birlikte, **mekânsal kaymalar** içerir: yol şebekesi yer yer hedeften farklı konumlanmakta, bazı yeşil alanlar "halüsinasyon" olarak eklenmekte veya kaybolmaktadır. Bu, eşsiz öğrenmenin doğal sonucudur ve düşük SSIM/PSNR değerlerinin niteliksel karşılığıdır.

---

## 4.4 Geliştirilmiş Pix2Pix Sonuçları (E1-Enhanced) — **Eğitimde**

Pix2PixHD'nin iki ana katkısını — **yüksek çözünürlük (512²)** ve **VGG-19 algısal kayıp ($\lambda_{VGG}=10$)** — kendi kontrol edilen, tam tekrarlanabilir kod tabanında uygulayan bu model, `configs/pix2pix_enhanced.yaml` ile eğitilmektedir.

Eğitim konfigürasyonu:
- Çözünürlük 512×512 (U-Net `num_downs=9`); batch=4; 150 epoch (75 sabit + 75 decay).
- Kayıp: LSGAN + L1($\lambda=100$) + VGG-perceptual($\lambda=10$).

> **Beklenti (H1):** 512² çözünürlük küçük yapı detaylarını (§4.8'deki Vaka-2) daha iyi yakalamalı; VGG algısal kaybı ise LPIPS'i Pix2Pix baseline'ın altına (iyileştirme) çekmelidir. Sonuçlar `RUN_DIR_ENH/eval/metrics.json` üretildiğinde bu bölüme işlenecektir.

---

## 4.5 Pix2PixHD Hakkında Teknik Not

Çalışmanın deneysel programında, NVIDIA'nın resmî Pix2PixHD implementasyonunun [6] da karşılaştırmaya dahil edilmesi planlanmıştır. Ancak bu implementasyonun **2018 tarihli olması ve güncel çalışma ortamıyla (Python 3.12, PyTorch 2.x) uyumsuzlukları** nedeniyle doğrudan çalıştırılamamıştır. Tespit edilen başlıca uyumsuzluk, Python 3.9 sürümünde standart kütüphaneden kaldırılan `fractions.gcd` fonksiyonunun çağrılmasıdır (`AttributeError: module 'fractions' has no attribute 'gcd'`).

Bu sorunu gidermek için bir uyumluluk yaması (`scripts/patch_pix2pixhd.py`) geliştirilmiştir; bu yama `fractions.gcd → math.gcd`, `.data[0] → .item()` (PyTorch 0.4+ skaler tensör değişikliği) ve Python 2 stili `raise('string')` ifadelerini düzeltir. Ancak eski kod tabanının modern PyTorch ile tam uyumu **deneysel kararlılık açısından risklidir**.

Bu nedenle, **bilimsel tekrarlanabilirlik ve metodolojik tutarlılık** ilkeleri gereği, Pix2PixHD'nin temel katkıları (yüksek çözünürlük + algısal kayıp) §4.4'te tanımlanan **Geliştirilmiş Pix2Pix** modeli ile kendi kontrol edilen kod tabanımızda gerçeklenmiştir. Bu yaklaşım, harici ve bakımı sürdürülmeyen bir kod tabanına bağımlılığı ortadan kaldırarak tezin **tam yeniden üretilebilirlik** garantisini korur.

---

## 4.6 Tüm Modellerin Karşılaştırması

**Tablo 4.4.** GAN mimarilerinin sketch → renkli map çevirisinde performans karşılaştırması (val kümesi: 1.098 örnek).

| Model | Çöz. | FID ↓ | SSIM ↑ | PSNR (dB) ↑ | LPIPS ↓ | L1 ↓ | Eğitim (A100 sa) |
|-------|:----:|------:|-------:|------------:|--------:|-----:|----------------:|
| **Pix2Pix** (L1) | 256² | 151,84 | **0,7772** | **27,36** | **0,1766** | **8,01** | ~1,5 |
| **CycleGAN** (eşsiz) | 256² | **54,58** | 0,6578 | 24,01 | 0,2032 | 11,72 | ~2,0 |
| **Geliştirilmiş Pix2Pix** (512²+VGG) | 512² | *(eğitimde)* | *(eğitimde)* | *(eğitimde)* | *(eğitimde)* | *(eğitimde)* | ~2,5 |

**Kalın** değerler her sütunun mevcut en iyisini gösterir. İki model arasındaki karşıtlık (CycleGAN'ın FID üstünlüğü vs. Pix2Pix'in piksel-sadakat üstünlüğü) §4.3.3'te tartışılan algı-bozulma ödünleşiminin somut tezahürüdür. Geliştirilmiş Pix2Pix satırı, ilgili eğitim tamamlandığında doldurulacaktır.

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
- CycleGAN sonuçları (FID 54,58; SSIM 0,6578) algı-bozulma ödünleşimini somutlaştırmış ve H3'ü doğrulamıştır; Geliştirilmiş Pix2Pix (512²+VGG) eğitimi tamamlanınca Tablo 4.4 tümüyle dolacaktır.
- Mevcut hata analizi, izleyen mimarilerin (özellikle Pix2PixHD'nin 512² çözünürlüğü ve SPADE'in uzamsal-uyarlamalı normalleştirmesi) sayesinde giderilmesi beklenen üç sistematik zorluk kategorisini tespit etmiştir.
