# GAN Urban Design v2
## GAN Modelleri Kullanarak Kentsel Peyzaj Tasarım Planlarının Otomatik Oluşturulması ve Renklendirilmesi

Fırat Üniversitesi Fen Bilimleri Enstitüsü Yüksek Lisans Tezi — Uygulama Reposu.

### Colab'da Hızlı Aç

| Notebook | Açıklama | Aç |
|----------|----------|----|
| `colab_pix2pix.ipynb` | Bizim Pix2Pix implementasyonu (sketch→map) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/gan-urban-design-v2/blob/main/notebooks/colab_pix2pix.ipynb) |
| `colab_external_models.ipynb` | CycleGAN + Pix2PixHD + Geliştirilmiş Pix2Pix karşılaştırma | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/gan-urban-design-v2/blob/main/notebooks/colab_external_models.ipynb) |
| `colab_figures.ipynb` | Veriye-bağlı tez şekilleri (dataset/niteliksel kıyaslama) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/gan-urban-design-v2/blob/main/notebooks/colab_figures.ipynb) |

> Badge çalışmazsa URL'i tarayıcıya yapıştırın:
> `https://colab.research.google.com/github/ilker-23/gan-urban-design-v2/blob/main/notebooks/<NOTEBOOK>.ipynb`

### Tez Görselleri Galerisi

Statik (veriye bağımsız) görseller `figures/` dizininde commit edildi:

| Şekil | Açıklama | Dosya |
|-------|----------|-------|
| 3.1 | Önerilen iki-aşamalı GAN pipeline'ı | `figures/fig_3_1_pipeline.png` |
| 3.2 | U-Net jeneratör mimarisi (256/512 görselleştirme) | `figures/fig_3_2_unet.png` |
| 3.3 | 70×70 PatchGAN ayırıcı | `figures/fig_3_3_patchgan.png` |
| 3.4 | SPADE blok diyagramı (denklem 3.9) | `figures/fig_3_4_spade.png` |
| 4.3 | 3 model × 5 metrik bar chart | `figures/fig_4_3_metrics_bars.png` |
| 4.4 | Radar grafiği (normalize profil) | `figures/fig_4_4_radar.png` |
| 4.5 | Enhanced vs baseline iyileşme yüzdeleri | `figures/fig_4_5_improvements.png` |
| 4.6 | Algı-bozulma ödünleşimi (FID×SSIM scatter) | `figures/fig_4_6_tradeoff.png` |

Bu görseller `python3 scripts/figures/make_static_figures.py` ile yeniden üretilebilir.

**Veriye bağımlı görseller** (4.1 dataset örnekleri, 4.2 sentetik sketch yöntemleri, 4.7 niteliksel model karşılaştırması, 4.8 eğitim progresi) için `colab_figures.ipynb` Colab notebook'unu çalıştırın — Drive'daki sample dosyalarını okur, `figures/data/` ve Drive'a yazar.

---

## Proje Özeti

İki aşamalı bir koşullu GAN pipeline'ı ile:
- **Aşama 1 — Plan/Layout üretimi:** Aerial görüntüden veya semantic mask'ten renk-kodlu kentsel plan üretimi.
- **Aşama 2 — Kroki renklendirme:** Siyah-beyaz krokiden (sketch) renkli, sunum-hazır plana çeviri.

Karşılaştırılan modeller: **Pix2Pix**, **CycleGAN**, **Pix2PixHD**, **SPADE/GauGAN**.

---

## Hızlı Başlangıç (Colab Pro A100)

```bash
# Colab notebook'unun ilk hücresinde:
!git clone https://github.com/<KULLANICI>/gan-urban-design-v2.git
%cd gan-urban-design-v2
!pip install -q -r requirements.txt
!bash scripts/download_data.sh         # maps dataset (~250 MB)
!python scripts/prepare_sketches.py    # sentetik sketch çiftlerini üret
!python -m src.train --config configs/pix2pix_default.yaml
```

Drive bağlama (checkpoint kaybını önler):
```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["RUN_DIR"] = "/content/drive/MyDrive/thesis_runs"
```

---

## Klasör Yapısı

```
gan-urban-design-v2/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                         # Araştırma raporu, dataset kararı
├── configs/                      # YAML eğitim ayarları
├── data/                         # Veri (gitignore — Drive/Colab'da tutulur)
├── scripts/                      # İndirme, sketch üretimi, vb. yardımcı scriptler
├── src/
│   ├── models/                   # UNet, PatchGAN, Pix2Pix wrapper
│   ├── datasets/                 # PyTorch Dataset sınıfları
│   ├── losses.py                 # GAN + L1 + perceptual kayıplar
│   ├── train.py                  # Ana eğitim girdisi
│   ├── evaluate.py               # FID, SSIM, PSNR, LPIPS hesaplama
│   └── utils.py                  # Yardımcılar
├── notebooks/                    # Colab notebook'ları
└── results/                      # Çıktılar, görüntüler, checkpoint'ler (gitignore)
```

---

## Datasetler

| Dataset | Boyut | Kaynak | Komut |
|---------|-------|--------|-------|
| **maps** (Pix2Pix Berkeley) | 2194 çift × 600² px | http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz | `bash scripts/download_data.sh maps` |
| **DeepGlobe Land Cover** | 1146 × 2448² px | Kaggle: `balraj98/deepglobe-land-cover-classification-dataset` | `bash scripts/download_data.sh deepglobe` |

Detaylar: [docs/02_Dataset_Karari.md](docs/02_Dataset_Karari.md).

---

## Modeller ve Beklenen Eğitim Süreleri (Colab Pro A100, 40 GB)

| Model | Çözünürlük | Epoch | Yaklaşık Süre |
|-------|-----------|-------|---------------|
| Pix2Pix (baseline) | 256² | 200 | ~1 saat |
| CycleGAN | 256² | 200 | ~2 saat |
| Pix2PixHD | 512² | 150 | ~4-5 saat |
| SPADE/GauGAN | 512² | 150 | ~5-7 saat |

---

## Değerlendirme Metrikleri

- **FID** (Fréchet Inception Distance) — dağılım benzerliği, ↓ düşük iyi
- **SSIM** — yapısal benzerlik, ↑ yüksek iyi
- **PSNR** — piksel düzeyinde kalite, ↑ yüksek iyi
- **LPIPS** — algısal benzerlik (VGG uzayı), ↓ düşük iyi
- **L1** — ortalama mutlak hata, ↓ düşük iyi

`python -m src.evaluate --checkpoint <path> --config <yaml>` ile hepsi tek seferde hesaplanır.

---

## Sonraki Adımlar

1. `scripts/download_data.sh` → datayı çek
2. `scripts/prepare_sketches.py` → sentetik sketch çiftlerini üret
3. `notebooks/01_dataset_eda.ipynb` → veriyi görselleştir
4. `notebooks/02_train_pix2pix.ipynb` → baseline'ı eğit
5. `notebooks/03_evaluate.ipynb` → tüm modelleri kıyasla

---

## Atıflar

- Isola vd. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*.
- Zhu vd. (2017). CycleGAN. *ICCV*.
- Wang vd. (2018). Pix2PixHD. *CVPR*.
- Park vd. (2019). SPADE/GauGAN. *CVPR*.
- Chen vd. (2024). Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches. *Land*, 13(2), 254.
- Jiang vd. (2024). Automated site planning using CAIN-GAN model. *Automation in Construction*.

Tam liste: [docs/01_Arastirma_Raporu.md](docs/01_Arastirma_Raporu.md).
