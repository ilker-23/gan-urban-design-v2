# DERİN ARAŞTIRMA RAPORU
## "GAN Modelleri Kullanarak Kentsel Peyzaj Tasarım Planlarının Otomatik Oluşturulması ve Renklendirilmesi"

**Hazırlayan:** Tez Asistanı (GAN uzmanı bakış açısıyla)
**Hedef kurum:** Fırat Üniversitesi, Fen Bilimleri Enstitüsü
**Tarih:** 31 Mayıs 2026
**Süre kısıtı:** 4-5 gün içerisinde uygulama + tez yazımı
**Donanım:** Google Colab Pro – NVIDIA A100 (40 GB)
**Kod barındırma:** GitHub → Colab `git clone` iş akışı

---

## 1. YÖNETİCİ ÖZETİ

Tez önerisi iki temel problem üzerine kuruludur:

1. **(P1) Plan Üretimi:** Şehir/park ölçeğinde kentsel peyzaj planlarının (özellikle park ve bahçelerin) **otomatik üretilmesi**.
2. **(P2) Renklendirme:** Mimar/peyzaj mimarının elle çizdiği **kroki/sketch** planların **renkli, sunum-hazır plana** dönüştürülmesi.

Literatür taraması, her iki problemin de **koşullu GAN (cGAN)** ailesi ile son 3 yılda çözüldüğünü göstermektedir. Tez önerisinde referans verilen **CAIN-GAN (Jiang vd., 2024, Elsevier – *Automation in Construction*)** P1 için bir devlet-of-the-art örneğidir; P2 için ise neredeyse birebir başlık olan **Chen vd. (2024, MDPI *Land*)** çalışması — "Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches" — temel kıyaslama (benchmark) yayını olarak kullanılmalıdır.

**Önerilen tez stratejisi:**
- **İki aşamalı pipeline** kurmak: Stage-1 (semantic layout → plan) + Stage-2 (kroki → renkli plan).
- **Baseline** olarak **Pix2Pix (Isola vd., 2017)** ve **CycleGAN (Zhu vd., 2017)**.
- **İleri model** olarak **Pix2PixHD (Wang vd., 2018)** veya **SPADE/GauGAN (Park vd., 2019, CVPR)**.
- **Dataset:** OSM (OpenStreetMap) park poligonları + uydu görüntüleri ile **özgün küçük dataset (~300–500 çift)** + **DeepGlobe Land Cover** açık dataset'i (1,146 görüntü, 2448×2448 px) ile ön-eğitim.
- **Metrikler:** FID, SSIM, PSNR, LPIPS, L1 (sektörde standart).

**4-5 gün** içerisinde Colab Pro A100 üzerinde Pix2Pix ailesi için 100–200 epoch (256×256–512×512 çözünürlükte) ulaşılabilir; Pix2PixHD/SPADE için 80–150 epoch realistiktir.

---

## 2. PROBLEMİN GAN PERSPEKTİFİNDEN KONUMLANDIRILMASI

### 2.1 GAN'lar Neden Bu Problem İçin Uygun?

Goodfellow vd. (2014) tarafından önerilen **Generative Adversarial Network (GAN)** mimarisi, generator G ve discriminator D ağlarının min-max oyunu ile gerçek dağılımı taklit etmeyi öğrenir:

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] $$

Tez konumuz **image-to-image translation** (görüntüden-görüntüye çeviri) alt problemine girer. Bu alt problem için altın standart **koşullu GAN (Mirza & Osindero, 2014)** ve onun image-to-image özelleşmesi olan **Pix2Pix (Isola vd., 2017, CVPR)**'dir. Tipik bir cGAN kaybı:

$$ \mathcal{L}_{cGAN}(G,D) = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,z}[\log(1-D(x,G(x,z)))] + \lambda \cdot \mathcal{L}_{L1}(G) $$

L1 kaybı, üretilen görüntünün hedef görüntüye **piksel-düzeyinde yakınlığını** zorlar; bu, plan/diyagram üretimi için kritiktir çünkü plan görüntülerinde **net sınırlar ve renk bölgeleri** vardır (doğal görüntü dağılımından farklıdır).

### 2.2 Bu Tezin Üç Alt-Görevi

| # | Alt-görev | Girdi | Çıktı | Uygun Model |
|---|-----------|-------|-------|-------------|
| T1 | **Site layout sentezi** | Şehir sınırı + kısıtlar (yol, su, mevcut yapı) | Park/bahçe yerleşim planı (semantic mask) | Pix2PixHD, CAIN-GAN, House-GAN++ |
| T2 | **Plan görselleştirme** | Semantic mask (renk kodlu bölgeler) | Foto-gerçekçi plan render | SPADE/GauGAN |
| T3 | **Sketch renklendirme** | Siyah-beyaz elle çizilmiş kroki | Renkli, sunum-hazır plan | Pix2Pix, CycleGAN, Chen vd. 2024 |

Tez önerisindeki "Şekil 2" → "Şekil 3" akışı **T3**'e karşılık geliyor; bu raporun ana odağı T3 olmakla birlikte T1+T3'ün **birleşik pipeline'ı** önerilmiştir.

---

## 3. DETAYLI LİTERATÜR TARAMASI

### 3.1 Temel (Foundational) GAN Modelleri

| # | Çalışma | Yıl | Yayın yeri | Özellik / İlgi |
|---|---------|-----|------------|----------------|
| R1 | Goodfellow vd., "Generative Adversarial Nets" | 2014 | NeurIPS | Orijinal GAN çerçevesi. |
| R2 | Mirza & Osindero, "Conditional GAN" | 2014 | arXiv | Koşullu üretim — tezin temel paradigması. |
| R3 | Radford vd., "DCGAN" | 2016 | ICLR | Conv-tabanlı stabil GAN eğitimi. |
| R4 | Isola vd., "Image-to-Image Translation with cGAN (Pix2Pix)" | 2017 | CVPR | **Baseline modelimiz.** Eşli (paired) çeviri. |
| R5 | Zhu vd., "CycleGAN" | 2017 | ICCV | **Baseline-2.** Eşsiz (unpaired) çeviri. |
| R6 | Wang vd., "Pix2PixHD" | 2018 | CVPR | Yüksek çözünürlüklü plan üretimi (2048×1024). |
| R7 | Park vd., "SPADE / GauGAN" | 2019 | CVPR (Oral) | **Semantic layout → foto-gerçekçi** çeviri; mekansal-uyarlamalı normalizasyon. |
| R8 | Karras vd., "StyleGAN2" | 2020 | CVPR | Gürültüden yüksek-kalite üretim (T1 alternatifi). |

### 3.2 Doğrudan İlgili: GAN + Kentsel/Peyzaj Planlama

| # | Çalışma | Yıl | Yayın | Yöntem | İlgi |
|---|---------|-----|-------|--------|------|
| U1 | **Chen vd., "Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches"** | 2024 | **MDPI *Land* 13(2):254**, DOI: 10.3390/land13020254 | cGAN (Pix2Pix vs CycleGAN), data augmentation, **152 eşli görsel** | **★ ANA BENCHMARK — birebir aynı problem.** |
| U2 | **Jiang vd., "Automated site planning using CAIN-GAN model"** | 2024 | **Elsevier *Automation in Construction* (S0926580524000220)** | Context-aware site planning GAN; NYC case study | **★ Tez önerisindeki referans (Şekil 1).** |
| U3 | Ye, Du, Ye, "MasterplanGAN: Facilitating the smart rendering of urban master plans via GANs" | 2022 | SAGE *EPB: Urban Analytics & City Science* | Pix2Pix tabanlı master-plan render | Render aşaması (T2) için. |
| U4 | "Suggestive Site Planning with Conditional GAN and Urban GIS Data" | 2019 | Springer (CDRF) | **Pix2PixHD**, OSM/GIS → site boundary → building footprint | GIS verisi pipeline için referans. |
| U5 | Chen vd., "UrbanGenoGAN: pioneering urban spatial planning using the synergistic integration of GAN, GA, and GIS" | 2023 | Frontiers in Env. Sci. | GAN + Genetik Algoritma + GIS | Optimizasyon eklemek isterseniz. |
| U6 | "Architectural layout generation using a graph-constrained cGAN" | 2023 | Elsevier *Automation in Construction* (S0926580523003138) | Graph-cGAN | Kısıt-tabanlı plan üretimi. |
| U7 | "Building layout generation using site-embedded GAN model" | 2023 | Elsevier *Automation in Construction* (S0926580523001486) | Site-gömülü cGAN | Site bağlamlı üretim. |
| U8 | "Automated site layout generation for buildings using graph-constrained GAN (GCGAN)" | 2025 | Springer *Building Simulation* (10.1007/s12273-025-1350-7) | GraphVAE + GAN | En güncel (2025) yayın. |
| U9 | "Automated building layout generation using deep learning and graph algorithms" | 2023 | Elsevier *Automation in Construction* (S0926580523002960) | Hibrit DL + graph | Karşılaştırma için. |
| U10 | Nauata vd., "House-GAN" / "House-GAN++" | 2020 / 2021 | ECCV / CVPR | Graf-kısıtlı ev planı | T1 için sağlam baseline. |
| U11 | "Automated urban landscape design: an AI-driven model for emotion-based layout generation and appraisal" | 2024 | PMC11623217 | Perception–reasoning–generation framework | Multi-objektif değerlendirme. |
| U12 | "Generative AI for Urban Design: A Stepwise Approach Integrating Human Expertise with Multimodal Diffusion Models" | 2025 | arXiv 2505.24260 | Diffusion alternatifi | GAN→Diffusion karşılaştırması (tartışmada). |
| U13 | "Combined Multi-Condition GAN and Force-Directed Algorithm for Generating Intelligent Residential Layout" | 2025 | Wiley *Transactions in GIS* | Multi-condition cGAN | Kısıt yönetimi. |
| U14 | "Generative Adversarial Networks in the built environment: A comprehensive review" | 2023 | Elsevier *Building & Environment* (S0360132322007089) | **Review makalesi** | Giriş bölümünde özetlenmeli. |
| U15 | "Automated layout generation from sites to flats using GAN and transfer learning" | 2023 | Elsevier (OUCI) | Transfer learning ile GAN | Veri-kısıtlı eğitim için. |
| U16 | "A Discussion on an Urban Layout Workflow Utilizing GAN — automatized labeling and dataset acquisition" | 2023 | Academia.edu | Etiketleme otomasyonu | Dataset üretimi için. |

### 3.3 Sketch / Sketch-to-Image Renklendirme (T3 için doğrudan kaynaklar)

| # | Çalışma | Yıl | Katkı |
|---|---------|-----|-------|
| S1 | Sangkloy vd., "Scribbler: Controlling Deep Image Synthesis with Sketch and Color" | 2017 | Sketch+renk ipucu → görüntü. |
| S2 | Chen & Hays, "SketchyGAN" | 2018 | 50K sketch-görüntü çiftiyle eğitim. |
| S3 | Liu vd., "Auto-Painter" | 2018 | Anime/illüstrasyon koloring; benzer mimari plana uyarlanabilir. |
| S4 | "Visual-attention GAN for interior sketch colorization" | 2021 | Dikkat mekanizmalı sketch → renkli. |
| S5 | Chen vd. (U1, 2024) | 2024 | Park sketch → renkli plan. **Tezin doğrudan kıyaslama referansı.** |

### 3.4 GAN'ı Şu An Önümüze Geçebilecek Alternatif: Difüzyon Modelleri

ControlNet (Zhang vd., 2023), Stable Diffusion (Rombach vd., 2022) ve InstructPix2Pix (Brooks vd., 2023) son 2 yılda GAN'ları image-to-image çevirisinde geçmiştir. **Tez konusu GAN'lara odaklı** olduğu için, **Tartışma** bölümünde difüzyon karşılaştırmasını "Gelecek çalışmalar" olarak konumlandırmak akademik olarak doğru olur — GAN-merkezli yazıyı zayıflatmaz, savunmada güçlendirir.

---

## 4. ANAHTAR MODELLERİN KARŞILAŞTIRMALI ANALİZİ

### 4.1 Pix2Pix (R4) — Önerilen Birincil Baseline
- **Mimari:** U-Net generator (skip-connections) + PatchGAN discriminator (70×70 patch).
- **Kayıp:** cGAN + L1 (λ=100).
- **Veri gereksinimi:** Eşli (paired) — sketch + renkli plan çifti.
- **Çözünürlük:** 256×256 (Colab'da rahat), 512×512 (A100'de yapılabilir).
- **Eğitim süresi (A100, 152 çift, 200 epoch, 256²):** ~45–60 dk.

### 4.2 CycleGAN (R5) — Eşsiz Veri için Alternatif
- **Avantaj:** Sketch ve renkli plan **eşli olmasa bile** çalışır.
- **Dezavantaj:** İçerik tutarlılığı Pix2Pix kadar yüksek değil; plan/diyagram için **piksel hassasiyetinden kaybeder**.
- **Tez için rolü:** "Eşli olmayan veri ile karşılaştırma" — savunmada güçlü argüman.

### 4.3 Pix2PixHD (R6) — Yüksek Çözünürlük için Önerilen Ana Model
- **Mimari:** Coarse-to-fine generator + multi-scale discriminator (3 ölçek) + perceptual (VGG) kayıp.
- **Çözünürlük:** 1024×512'ye kadar uygulanabilir.
- **A100'de feasible:** 200 epoch, ~3–5 saat.

### 4.4 SPADE / GauGAN (R7) — Semantic Layout için Önerilen İleri Model
- **Anahtar yenilik:** SPADE normalizasyonu, semantic mask'in bilgisinin derin katmanlarda "yıkanıp gitmesini" önler.
- **Tez için kritik:** "Renk kodlu plan bölgeleri" tam olarak semantic mask'tir; SPADE bu problem için **Pix2PixHD'den daha uygun**.
- **GauGAN demo:** NVIDIA'nın landscape image üretimi için eğitilmiş versiyonu — peyzaj domain'iyle birebir örtüşür.

### 4.5 CAIN-GAN (U2) — Tez Önerisinin Referans Modeli
- **Açılım:** Context-Aware Site Planning GAN.
- **İki aşama:** (Phase 1) Footprint construction → (Phase 2) Height completion.
- **Yenilik:** Domain knowledge gömülü (urban sustainability metrikleri).
- **Tez için rolü:** Önerilen mimari **CAIN-GAN'ın peyzaj/park'a uyarlanmış varyantı** olarak konumlandırılabilir; bu, hem özgünlük hem akademik altyapı sağlar.

### 4.6 Karşılaştırma Tablosu

| Model | Veri Tipi | Çözünürlük | Eğitim (A100) | Tez için Rol |
|-------|-----------|------------|---------------|--------------|
| Pix2Pix | Paired | 256–512 | 1 saat | **Baseline** |
| CycleGAN | Unpaired | 256 | 2 saat | Karşılaştırma |
| Pix2PixHD | Paired | 1024 | 4–5 saat | **Önerilen-1** |
| SPADE/GauGAN | Paired (mask) | 512–1024 | 5–7 saat | **Önerilen-2** |
| CAIN-GAN | Paired + bağlam | 512 | (özelleşmiş) | Akademik referans |

---

## 5. VERİ SETİ ÖNERİLERİ

### 5.1 Halka Açık Datasetler (Hazır indir-eğit)

| Dataset | Boyut | İçerik | Lisans | İlgi düzeyi |
|---------|-------|--------|--------|-------------|
| **DeepGlobe Land Cover** (Kaggle) | 1,146 × 2448² px | 7 sınıf (urban, agriculture, rangeland, **forest**, water, barren, unknown) | Açık (yarışma) | ★★★ Park/yeşil alan segmentasyon ön-eğitimi için. |
| **CMP Facades** | 400 | Semantic label → bina cephesi | Open | ★★ Pix2Pix tutorial veri (ön-eğitim/sanity check). |
| **Cityscapes** | 2,975 | Şehir görüntüsü + semantic mask | Akademik (kayıt) | ★★ SPADE eğitimi için klasik. |
| **House-GAN dataset** | ~143K | Vektör floor plan | Açık | ★★ İç mekan ama mimari layout GAN için iyi. |
| **MSD (Modified Swiss Dwellings)** | Geniş | Multi-apartment floorplan | Kaggle açık | ★ Layout generation. |
| **ResPlan** | 17,000 | Vector-graph residential | arXiv 2508.14006 | ★ Yeni (2025). |

### 5.2 Önerilen Özgün Dataset (Tez Özgün Katkısı için Kritik)

**Strateji:** OpenStreetMap (OSM) park/bahçe poligonları + uydu/aerial görüntü eşlemesi.

**Pipeline:**
1. **OSM Overpass API** ile İstanbul, Ankara, İzmir, Elazığ (Fırat'ın bulunduğu il – yerel bağlam) gibi şehirlerden `leisure=park` ve `leisure=garden` poligonları çek.
2. Her park için **Mapbox Static Images API** veya **Google Earth Engine** ile yüksek çözünürlüklü orthofoto al.
3. OSM içindeki alt-poligonları (yol, su, ağaç, bina, spor alanı) **renk-kodlu semantic mask**'e dönüştür.
4. Renkli orthofoto'dan **sketch'e çevirim** (Canny edge / XDoG filter ile **otomatik sentetik sketch** üretimi) → eşli çift elde et.

**Hedef dataset boyutu:** 300–600 eşli (sketch, semantic_mask, color_render) üçlüsü.

**Veri augmentation:**
- Rotasyon (0°, 90°, 180°, 270°)
- Yatay/dikey flip
- Random crop (256×256 patch'ler)
- Color jitter (renkli plan tarafında)

Bu, Chen vd. (2024)'ün 152 çiftine kıyasla **3-4 kat daha büyük dataset** anlamına gelir — tezin **özgünlük** maddesi için yeterli.

### 5.3 Hazır Bir Sketch→Plan Dataset'i Yoksa Ne Yapacağız?

**Mevcut durumda halka açık, hazır bir "park sketch → renkli plan" dataset'i yok.** Bu, Chen vd. (2024)'ün de işaret ettiği bir boşluk. Bu **tezin özgün katkısının ana kalemlerinden biri** olabilir: *"Park-sketch-color rendering için ilk halka açık dataset'in oluşturulması."*

---

## 6. ÖNERİLEN YÖNTEM VE MİMARİ

### 6.1 İki Aşamalı Pipeline Önerisi

```
                ┌─────────────────────────┐                ┌─────────────────────────┐
   şehir        │  Stage-1: Layout Gen.   │   semantic     │  Stage-2: Render/Color  │
   sınırı  ───► │  (Pix2PixHD veya CAIN-  │   mask + ───►  │  (SPADE/GauGAN veya     │   renkli
   + GIS        │   GAN benzeri varyant)  │   sketch       │   Pix2Pix)              │   plan
                └─────────────────────────┘                └─────────────────────────┘
```

### 6.2 Stage-1: Plan Üretimi (T1)
- **Girdi:** 256×256 veya 512×512 görüntüde park sınırı (binary mask) + yol/su gibi context.
- **Çıktı:** Renk-kodlu semantic mask (yeşil=çim, koyu yeşil=ağaç, mavi=su, gri=yol, kahverengi=oturma, vs. — **~8 sınıf**).
- **Model:** Pix2PixHD (U-Net + multi-scale D).
- **Kayıp:** cGAN + L1(λ=100) + VGG-perceptual(λ=10).

### 6.3 Stage-2: Sketch→Renkli Plan (T3 — tezin ana iddiası)
- **Girdi:** Siyah-beyaz kroki (gerçek elle çizilmiş VEYA Stage-1 çıktısının edge'e çevirisinden sentetik).
- **Çıktı:** Renkli, sunum-hazır plan.
- **Model:** Pix2Pix (baseline) + SPADE (ana model).
- **Karşılaştırma:** Chen vd. (2024) ile **aynı metriklerle** doğrudan kıyaslama.

### 6.4 Önerilen Özgün Katkılar
1. **İki-aşamalı joint pipeline** — literatürde T1 ve T3 ayrı çalışılmış, birleştiren çalışma minimal.
2. **OSM tabanlı yeni dataset** — Türkiye şehirlerinden park örneklerini de içeren ilk açık dataset.
3. **SPADE'in park-sketch domain'ine adaptasyonu** — Chen vd. (2024) sadece Pix2Pix/CycleGAN denemiş; SPADE denenmemiş.
4. **Sentetik sketch üretimi (XDoG + augmentation)** ile veri ölçeklendirme.
5. **Quantitative + qualitative** birleşik değerlendirme: FID, SSIM, PSNR, LPIPS + insan-uzmanı (peyzaj mimarı) anket skoru.

### 6.5 Değerlendirme Protokolü

| Metrik | Ne ölçer | Hedef yön |
|--------|----------|-----------|
| **FID** (Fréchet Inception Distance) | Dağılım benzerliği | ↓ düşük iyi |
| **SSIM** | Yapısal benzerlik | ↑ yüksek iyi |
| **PSNR** | Piksel düzeyinde kalite | ↑ yüksek iyi |
| **LPIPS** | Algısal benzerlik (VGG-uzayı) | ↓ düşük iyi |
| **L1** | Ortalama mutlak hata | ↓ düşük iyi |
| **Uzman skoru** | Peyzaj mimarı 1-5 Likert | ↑ yüksek iyi |

**Bölme:** %70 train / %15 val / %15 test.
**Tekrarlanabilirlik:** Sabit seed (42), Colab notebook'lar GitHub'da, modeller Drive'da.

---

## 7. UYGULAMA ALTYAPISI — COLAB PRO A100 + GITHUB İŞ AKIŞI

### 7.1 Repo İskeleti
```
gan-urban-design-v2/
├── README.md
├── requirements.txt
├── configs/
│   ├── pix2pix_base.yaml
│   ├── pix2pixhd.yaml
│   └── spade.yaml
├── data/
│   ├── README.md           # Drive'dan indirme talimatı
│   └── prepare_osm.py      # OSM→dataset oluşturma
├── src/
│   ├── models/
│   │   ├── pix2pix.py
│   │   ├── pix2pixhd.py
│   │   ├── spade.py
│   │   └── cyclegan.py
│   ├── datasets/
│   │   ├── park_sketch.py
│   │   └── deepglobe.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_pix2pix.ipynb
│   ├── 03_train_pix2pixhd.ipynb
│   ├── 04_train_spade.ipynb
│   ├── 05_evaluate.ipynb
│   └── 06_inference_demo.ipynb
├── scripts/
│   ├── download_data.sh
│   └── generate_sketches.py
└── results/
    └── (gitignore — Drive'a yazılır)
```

### 7.2 Colab Pro A100 İş Akışı
```python
# Notebook başına yerleştirilecek standart blok:
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<kullanici>/gan-urban-design-v2.git
%cd gan-urban-design-v2
!pip install -q -r requirements.txt
!nvidia-smi   # A100 doğrulama

# Checkpoint/log Drive'a:
import os
os.environ["RUN_DIR"] = "/content/drive/MyDrive/thesis_runs/spade_v1"
```

### 7.3 Önerilen Kütüphaneler
- **PyTorch 2.x** + **torchvision**
- `pytorch-fid` (FID metriği)
- `lpips` (LPIPS metriği)
- `scikit-image` (SSIM/PSNR)
- `albumentations` (veri augmentation)
- `wandb` veya `tensorboard` (eğitim takibi)
- `osmnx`, `geopandas`, `rasterio` (OSM/GIS veri toplama)
- Mevcut repo: **junyanz/pytorch-CycleGAN-and-pix2pix** ve **NVlabs/SPADE** referans alınmalı.

### 7.4 Eğitim Bütçesi (A100 üzerinde tahmini süreler)

| Aşama | Çözünürlük | Epoch | Tahmini Süre |
|-------|-----------|-------|--------------|
| Pix2Pix (baseline) | 256² | 200 | 1 sa |
| CycleGAN (karşılaştırma) | 256² | 200 | 2 sa |
| Pix2PixHD | 512² | 150 | 4–5 sa |
| SPADE/GauGAN | 512² | 150 | 5–7 sa |
| Toplam | | | ~12–15 sa GPU |

A100 saat ücreti hesabıyla 4 günde rahatlıkla 4 modeli de eğitebilirsiniz.

---

## 8. 4-5 GÜNLÜK ÇALIŞMA TAKVİMİ

| Gün | Sabah (4 sa) | Öğleden Sonra (4 sa) | Akşam (3 sa) |
|-----|--------------|----------------------|--------------|
| **Gün 1** | Literatür özetlerini topla, tez şablonu (Fırat FBE) indir, **Bölüm 1-2 (Giriş + Literatür)** yazımına başla | OSM/Mapbox API key'i al, **dataset toplama script'i** (`prepare_osm.py`) yaz | İlk 100 park örneğini topla, kontrol et |
| **Gün 2** | Dataset'i 300-500 çifte tamamla, augmentation pipeline'ı kur | **GitHub repo'yu kur**, Pix2Pix base kodunu adapte et, Colab notebook'larını yaz | Pix2Pix baseline eğitimi başlat (1 sa A100), sonuç görüntüle |
| **Gün 3** | **Pix2PixHD** eğitimi (4–5 sa A100), eş zamanlı **Bölüm 3 (Yöntem)** yazımı | **SPADE/GauGAN** eğitimi (5–7 sa A100), arka planda Bölüm 3 tamamla | CycleGAN eğitimi (karşılaştırma), ön-sonuçları topla |
| **Gün 4** | Tüm modellerin **test/inference** sonuçlarını üret, **metrikleri hesapla** (FID, SSIM, PSNR, LPIPS) | **Bölüm 4 (Deneysel Sonuçlar)** yazımı, görsel kıyaslama tabloları | **Bölüm 5 (Tartışma)** + Sonuç yazımı |
| **Gün 5** | Kaynakça, özetler (TR+EN), şekil-tablo numaralandırma | Final düzelti, intihal kontrolü, formatlama | Danışmana teslim |

---

## 9. FIRAT ÜNİVERSİTESİ FBE TEZ İSKELETİ

Tipik Türkiye yüksek lisans tezi yapısına göre:

```
ÖNSÖZ
ÖZET
ABSTRACT
İÇİNDEKİLER / ŞEKİLLER / TABLOLAR / SİMGELER

1. GİRİŞ
   1.1 Problemin Tanımı (sürdürülebilir kalkınma, kentsel zorluklar)
   1.2 Tezin Amacı ve Kapsamı
   1.3 Tezin Özgün Katkıları (5 madde — Bölüm 6.4)
   1.4 Tezin Organizasyonu

2. LİTERATÜR ARAŞTIRMASI
   2.1 Üretken Çekişmeli Ağlar — Temel
   2.2 Görüntüden-Görüntüye Çeviri (Pix2Pix, CycleGAN, Pix2PixHD)
   2.3 Semantic Image Synthesis (SPADE/GauGAN)
   2.4 GAN ile Otomatik Plan Üretimi (CAIN-GAN, MasterplanGAN, House-GAN++, UrbanGenoGAN)
   2.5 Sketch-to-Image Renklendirme (Chen vd. 2024 detaylı)
   2.6 Literatürdeki Boşluk ve Tezin Konumu

3. MATERYAL VE YÖNTEM
   3.1 Veri Seti
       3.1.1 DeepGlobe Land Cover ile Ön-eğitim
       3.1.2 OSM Tabanlı Özgün Park Dataset'i
       3.1.3 Sentetik Sketch Üretimi (XDoG)
       3.1.4 Veri Augmentation Stratejisi
   3.2 Önerilen İki-Aşamalı Pipeline
       3.2.1 Stage-1: Layout Generation (Pix2PixHD)
       3.2.2 Stage-2: Sketch-to-Render (SPADE)
   3.3 Kayıp Fonksiyonları
   3.4 Eğitim Detayları (Colab A100, hyperparam'lar)
   3.5 Değerlendirme Metrikleri (FID, SSIM, PSNR, LPIPS)

4. DENEYSEL SONUÇLAR
   4.1 Deney Düzeneği
   4.2 Baseline Karşılaştırma (Pix2Pix vs CycleGAN)
   4.3 İleri Model Sonuçları (Pix2PixHD, SPADE)
   4.4 Chen vd. 2024 ile Doğrudan Kıyaslama
   4.5 Ablation Studies (augmentation, kayıp katsayıları)
   4.6 Görsel/Niteliksel Değerlendirme
   4.7 Uzman Görüşü Anketi (varsa)

5. TARTIŞMA
   5.1 Bulguların Yorumlanması
   5.2 Sınırlamalar (veri boyutu, çözünürlük, domain shift)
   5.3 GAN vs Diffusion — Gelecek Yönelim

6. SONUÇLAR VE ÖNERİLER

KAYNAKLAR
EKLER (kod linki, ek görsel sonuçlar)
ÖZGEÇMİŞ
```

---

## 10. RİSKLER VE AÇIK SORULAR

| Risk | Etki | Azaltma |
|------|------|---------|
| Eşli (sketch-renkli) dataset elde etmek zor | Yüksek | XDoG + Canny ile **sentetik sketch** üretmek; gerçek elle-çizimi sadece test seti için 10-20 adet eklemek |
| 4-5 gün veri toplama+eğitim+yazıya yetmeyebilir | Yüksek | DeepGlobe + sentetik veri üzerinde başla, OSM-özgün dataset paralel |
| Mode collapse / unstable training | Orta | Pix2Pix temelli (stabil); LR=2e-4, Adam β1=0.5; spectral norm denenebilir |
| Colab oturum kopması | Orta | Checkpoint'leri Drive'a, her epoch'ta resume mümkün |
| Tez savunmasında "neden GAN, neden diffusion değil?" | Düşük | Tartışma bölümünde proaktif yanıt: tez konusu önerisi GAN-merkezli, diffusion gelecek çalışma; ayrıca GAN inference hızı (real-time) üstünlüğü |
| OSM verisi Türkiye'de seyrek | Orta | Yurtdışı parklar da dahil edilebilir; alternatif: önceden işlenmiş park görsellerini kullan |

---

## 11. KAYNAKLAR (Numaralı — Tez Bölüm 2'de Kullanılacak)

> Aşağıdaki kaynaklar IEEE/APA formatına çevrilerek tezin "Kaynaklar" bölümüne eklenmelidir. URL'ler ön-arama içindir; tezin son halinde DOI ile birlikte yazılmalıdır.

**Temel GAN:**
1. Goodfellow, I. vd. (2014). Generative Adversarial Nets. *NeurIPS*.
2. Mirza, M. & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv:1411.1784*.
3. Radford, A. vd. (2016). Unsupervised Representation Learning with DCGANs. *ICLR*.
4. Isola, P. vd. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*.
5. Zhu, J.-Y. vd. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *ICCV*.
6. Wang, T.-C. vd. (2018). High-Resolution Image Synthesis (Pix2PixHD). *CVPR*.
7. Park, T. vd. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE/GauGAN). *CVPR*.
8. Karras, T. vd. (2020). StyleGAN2. *CVPR*.

**GAN + Kentsel/Peyzaj Planlama (Tezin doğrudan kullanacağı kaynaklar):**
9. Chen, R. vd. (2024). Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches. *Land*, 13(2), 254. DOI:10.3390/land13020254.
10. Jiang, F. vd. (2024). Automated site planning using CAIN-GAN model. *Automation in Construction*. DOI:10.1016/j.autcon.2024.105306 (S0926580524000220).
11. Ye, X., Du, J., Ye, Y. (2022). MasterplanGAN: Facilitating the smart rendering of urban master plans via generative adversarial networks. *Environment and Planning B: Urban Analytics and City Science*.
12. Quan, S. (2019). Suggestive Site Planning with Conditional GAN and Urban GIS Data. *CDRF / Springer*.
13. Chen, R. vd. (2023). UrbanGenoGAN: pioneering urban spatial planning using the synergistic integration of GAN, GA, and GIS. *Frontiers in Environmental Science*.
14. Nauata, N. vd. (2020). House-GAN: Relational GAN for Graph-constrained House Layout Generation. *ECCV*.
15. Nauata, N. vd. (2021). House-GAN++: Generative Adversarial Layout Refinement Networks. *CVPR*.
16. (Yazar). (2023). Architectural layout generation using a graph-constrained conditional GAN. *Automation in Construction* (S0926580523003138).
17. (Yazar). (2023). Building layout generation using site-embedded GAN model. *Automation in Construction* (S0926580523001486).
18. (Yazar). (2025). Automated site layout generation for buildings using graph-constrained GAN. *Building Simulation*. DOI:10.1007/s12273-025-1350-7.
19. (Yazar). (2023). GANs in the built environment: A comprehensive review. *Building and Environment* (S0360132322007089).
20. (Yazar). (2024). Automated urban landscape design: an AI-driven model for emotion-based layout generation. *PMC11623217*.

**Sketch / Renklendirme:**
21. Sangkloy, P. vd. (2017). Scribbler: Controlling Deep Image Synthesis with Sketch and Color. *CVPR*.
22. Chen, W. & Hays, J. (2018). SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis. *CVPR*.

**Dataset/altyapı:**
23. Demir, I. vd. (2018). DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images. *CVPR Workshops*.
24. Cordts, M. vd. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. *CVPR*.
25. Tylecek, R. & Sara, R. (2013). Spatial Pattern Templates for Recognition of Objects with Regular Structure (CMP Facades). *GCPR*.

**Difüzyon (Tartışmada):**
26. Rombach, R. vd. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*.
27. Zhang, L. vd. (2023). Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet). *ICCV*.

---

## 12. SONRAKİ ADIM ÖNERİSİ

Bu rapor onaylandıktan sonra önereceğim sonraki adım:

1. **GitHub repo'sunu birlikte kurmak** (boş iskelet, README, requirements.txt, ilk Colab notebook).
2. **`prepare_osm.py` script'ini yazmak** (OSM Overpass + Mapbox veya alternatif görüntü kaynağıyla dataset toplamak).
3. **Pix2Pix baseline'ı Colab A100 üzerinde DeepGlobe'da eğitmek** (sanity check, ~1 saat).
4. Paralelde **Bölüm 1 (Giriş)** ve **Bölüm 2 (Literatür)** yazımına başlamak.

Hazır olduğunuzda **"hadi repo'yu kuralım"** dediğinizde Adım 1 ile başlayacağız.
