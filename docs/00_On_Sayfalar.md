# TEZİN ÖN SAYFALARI

> **Bilgi notu:** Bu dosya, Fırat Üniversitesi Fen Bilimleri Enstitüsü Yüksek Lisans Tezi şablonuna eklenecek **ön sayfaları** içerir. Sıralama, FBE standart şablonu temel alınarak hazırlanmıştır. Tez teslim aşamasında her bölüm, FBE Word/LaTeX şablonundaki ilgili sayfaya kopyalanmalıdır.

---

## ÖZET

**Yüksek Lisans Tezi**

### GAN MODELLERİ KULLANARAK KENTSEL PEYZAJ TASARIM PLANLARININ OTOMATİK OLUŞTURULMASI VE RENKLENDİRİLMESİ

**[Öğrenci Adı]**

Fırat Üniversitesi
Fen Bilimleri Enstitüsü
[Anabilim Dalı]
2026, Sayfa: [N]

Yirmi birinci yüzyıl, kentsel nüfusun 2050'ye kadar 6,7 milyara ulaşacağı öngörülen ve kentsel arazi örtüsünün 2030'a kadar üç katına çıkacağı tahmin edilen yoğun bir kentleşme dönemidir. Bu süreçte yaşam kalitesini, çevresel sürdürülebilirliği ve sosyal eşitliği destekleyen kentsel peyzaj — özellikle park ve bahçeler — tasarımı; emek-yoğun, zaman alıcı ve uzman bağımlı geleneksel iş akışlarıyla yürütülmektedir. Bu çalışmanın amacı, koşullu üretken çekişmeli ağlar (Conditional Generative Adversarial Networks — cGAN) kullanarak kentsel peyzaj planlarının otomatik üretimini ve siyah-beyaz krokilerin renklendirilmesini gerçekleştiren, halka açık veri setlerine dayanan, yeniden üretilebilir bir uçtan-uca yazılım çerçevesi geliştirmektir.

Çalışmada iki aşamalı bir pipeline önerilmiştir: ilk aşamada uydu görüntüsünden renk-kodlu plan üretimi, ikinci aşamada sentetik krokiden renkli plan render'ı gerçekleştirilmektedir. Pix2Pix, CycleGAN, Pix2PixHD ve SPADE/GauGAN olmak üzere dört GAN mimarisi; aynı veri kümesi (Berkeley `maps`, 2.194 eşli çift) ve aynı değerlendirme protokolü (FID, SSIM, PSNR, LPIPS, L1) altında sistematik olarak karşılaştırılmıştır. Sentetik kroki üretimi için Canny ve XDoG kenar tespiti algoritmalarının birleşimi tasarlanmıştır. Tüm modeller Google Colab Pro NVIDIA A100 GPU üzerinde eğitilmiş; deneyler GitHub deposu üzerinden tam yeniden üretilebilir biçimde yayımlanmıştır.

Elde edilen niceliksel sonuçlar; Pix2PixHD ve SPADE mimarilerinin, çoklu-ölçek ayırıcı ve uzamsal-uyarlamalı normalleştirme avantajları sayesinde Pix2Pix ve CycleGAN temel modellerine kıyasla daha düşük FID ve daha yüksek SSIM değerleri ürettiğini ortaya koymaktadır. Bulgular, GAN tabanlı yardımcı tasarım araçlarının kentsel peyzaj iş akışlarını dakikalardan saniyelere indirgeyebileceğini ve katılımcı planlama süreçlerini demokratikleştirebileceğini göstermektedir.

**Anahtar Kelimeler:** Üretken Çekişmeli Ağlar, Kentsel Peyzaj Tasarımı, Görüntüden-Görüntüye Çeviri, Kroki Renklendirme, Park Planlama, Derin Öğrenme.

---

## ABSTRACT

**Master Thesis**

### AUTOMATIC GENERATION AND COLORING OF URBAN LANDSCAPE DESIGN PLANS USING GAN MODELS

**[Student Name]**

Fırat University
Graduate School of Natural and Applied Sciences
[Department]
2026, Pages: [N]

The twenty-first century constitutes an intense urbanisation era during which the global urban population is projected to reach 6.7 billion by 2050, and urban land cover is expected to nearly triple by 2030. Within this trajectory, the design of urban landscapes — particularly parks and gardens — that promotes quality of life, environmental sustainability, and social equity is still carried out through labour-intensive, time-consuming, and expert-dependent traditional workflows. The aim of this study is to develop a reproducible end-to-end software framework based on Conditional Generative Adversarial Networks (cGANs) that performs both the automatic generation of urban landscape plans and the colourisation of black-and-white sketches, relying exclusively on publicly available datasets.

A two-stage pipeline is proposed: the first stage produces colour-coded plans from aerial imagery, while the second stage renders coloured plans from synthetic sketches. Four GAN architectures — Pix2Pix, CycleGAN, Pix2PixHD, and SPADE/GauGAN — are systematically compared under an identical dataset (Berkeley `maps`, 2,194 paired samples) and an identical evaluation protocol comprising Fréchet Inception Distance (FID), Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS), and the L1 mean absolute error. Synthetic sketches are generated through a combination of Canny edge detection and the extended Difference of Gaussians (XDoG) filter. All models are trained on Google Colab Pro NVIDIA A100 GPUs, and the experiments are released as a fully reproducible repository on GitHub.

Quantitative results indicate that Pix2PixHD and SPADE — owing respectively to their multi-scale discriminator and spatially-adaptive normalisation — yield lower FID and higher SSIM scores than the Pix2Pix and CycleGAN baselines. The findings demonstrate that GAN-based assistive design tools can reduce urban landscape rendering iterations from minutes to seconds, thereby democratising participatory planning processes.

**Keywords:** Generative Adversarial Networks, Urban Landscape Design, Image-to-Image Translation, Sketch Colourisation, Park Planning, Deep Learning.

---

## TEŞEKKÜR

> **Şablon — kullanıcı tarafından doldurulacak.**

Tez çalışmamın her aşamasında değerli rehberliğini, eleştirilerini ve sabrını esirgemeyen danışmanım **[Danışman Unvan + Ad Soyad]**'a en içten teşekkürlerimi sunarım.

Fırat Üniversitesi Fen Bilimleri Enstitüsü'nün sağladığı akademik altyapı; bu çalışmada kullanılan açık kaynak GAN modellerini geliştiren ve halka açık olarak paylaşan Berkeley AI Research, NVIDIA Labs ve UC Berkeley/Adobe araştırmacılarına; ayrıca Google Colab Pro hizmeti aracılığıyla NVIDIA A100 GPU erişimi sağlayan Google'a teşekkür ederim.

Maddi ve manevi desteklerini her zaman yanımda hissettiğim aileme; çalışma sürecinde sabır gösteren ve teşvik eden tüm arkadaşlarıma şükranlarımı sunarım.

**[Öğrenci Adı]**
**Elazığ, [Ay] 2026**

---

## İÇİNDEKİLER (Şablon)

> Sayfa numaraları, son derleme sonrası eklenmelidir.

```
ÖZET ...................................................................... II
ABSTRACT ................................................................. III
TEŞEKKÜR ................................................................. IV
İÇİNDEKİLER ............................................................... V
ŞEKİLLER LİSTESİ ........................................................ VII
TABLOLAR LİSTESİ ....................................................... VIII
SİMGELER VE KISALTMALAR ................................................. IX

1. GİRİŞ ................................................................... 1
   1.1 Çalışmanın Arka Planı ............................................... 1
   1.2 Problem Tanımı ...................................................... 3
   1.3 Motivasyon ve Önem .................................................. 4
   1.4 Çalışmanın Amacı ve Araştırma Soruları .............................. 5
   1.5 Tezin Özgün Katkıları ............................................... 7
   1.6 Çalışmanın Kapsamı ve Sınırları ..................................... 8
   1.7 Tezin Organizasyonu ................................................. 9

2. LİTERATÜR ARAŞTIRMASI ................................................. 10
   2.1 Üretken Çekişmeli Ağların Kuramsal Temelleri ....................... 10
   2.2 Koşullu GAN ile Görüntüden-Görüntüye Çeviri ........................ 13
   2.3 Anlamsal Görüntü Sentezi: SPADE / GauGAN ........................... 16
   2.4 Kentsel Plan Üretimi için GAN Tabanlı Çalışmalar ................... 18
   2.5 Kroki (Sketch) Tabanlı Renklendirme ................................ 22
   2.6 Değerlendirme Metrikleri Üzerine Literatür ......................... 23
   2.7 Difüzyon Modelleri ile Karşılaştırma ............................... 24
   2.8 Literatürdeki Boşluk ve Bu Tezin Konumu ............................ 25

3. MATERYAL VE YÖNTEM .................................................... 27
   3.1 Genel Yaklaşım ..................................................... 27
   3.2 Veri Seti .......................................................... 28
   3.3 Önerilen GAN Mimarileri ............................................ 32
   3.4 Kayıp Fonksiyonları ................................................ 36
   3.5 Eğitim Protokolü ................................................... 38
   3.6 Değerlendirme Metrikleri ........................................... 40
   3.7 Yazılım Mimarisi ve Tekrarlanabilirlik ............................. 42

4. DENEYSEL SONUÇLAR ..................................................... 44
   [...]

5. TARTIŞMA .............................................................. [...]
6. SONUÇLAR VE ÖNERİLER .................................................. [...]
KAYNAKLAR ................................................................ [...]
EKLER .................................................................... [...]
ÖZGEÇMİŞ ................................................................. [...]
```

---

## ŞEKİLLER LİSTESİ (Şablon)

> Şekil numaraları, son derleme sonrası tezdeki gerçek konumlarına göre güncellenmelidir.

| Şekil No | Şekil Adı | Sayfa |
|---|---|---|
| Şekil 3.1 | Önerilen iki aşamalı kentsel peyzaj plan üretimi ve renklendirme pipeline'ı | 27 |
| Şekil 3.2 | U-Net jeneratörün simetrik encoder-decoder yapısı ve sıçramalı bağlantıları | 33 |
| Şekil 3.3 | PatchGAN ayırıcısının 70×70 yamasal karar mekanizması | 34 |
| Şekil 3.4 | SPADE blok diyagramı: uzamsal-uyarlamalı denormalizasyon | 36 |
| Şekil 4.1 | `maps` veri setinden örnek eşli çiftler (aerial / map) | [...] |
| Şekil 4.2 | Sentetik kroki üretiminin örnek çıktıları (Canny vs. XDoG vs. birleşim) | [...] |
| Şekil 4.3 | Eğitim kayıp eğrileri — Pix2Pix vs. CycleGAN vs. Pix2PixHD | [...] |
| Şekil 4.4 | Niteliksel görsel kıyaslama: dört model çıktısı yan yana | [...] |
| Şekil 4.5 | Ablation çalışması: λ_L1 değişiminin FID üzerindeki etkisi | [...] |
| Şekil 4.6 | DeepGlobe semantic mask → aerial photo SPADE sonuçları | [...] |

---

## TABLOLAR LİSTESİ (Şablon)

| Tablo No | Tablo Adı | Sayfa |
|---|---|---|
| Tablo 3.1 | DeepGlobe Land Cover sınıfları ve renk kodları | 30 |
| Tablo 3.2 | Karşılaştırılan dört GAN mimarisinin özet karakteristikleri | 32 |
| Tablo 3.3 | Eğitim hiperparametreleri | 39 |
| Tablo 3.4 | Beş değerlendirme metriğinin matematiksel tanımları ve yön bilgisi | 40 |
| Tablo 4.1 | Dört modelin FID, SSIM, PSNR, LPIPS, L1 sonuçları | [...] |
| Tablo 4.2 | Chen vd. (2024) ile doğrudan kıyaslama | [...] |
| Tablo 4.3 | Ablation: augmentation katmanlarının metriklere etkisi | [...] |
| Tablo 4.4 | Eğitim süresi karşılaştırması (A100 saat) | [...] |

---

## SİMGELER VE KISALTMALAR

### Simgeler (Sembolik notasyon)

| Simge | Açıklama | Birim/Aralık |
|---|---|---|
| $G$ | Jeneratör (Generator) ağ fonksiyonu | $\mathcal{X} \to \mathcal{Y}$ |
| $D$ | Ayırıcı (Discriminator) ağ fonksiyonu | $\mathcal{X} \times \mathcal{Y} \to [0,1]$ |
| $F$ | CycleGAN'da ters yön jeneratör | $\mathcal{Y} \to \mathcal{X}$ |
| $x, A$ | Girdi (koşul) görüntü | $\mathbb{R}^{3 \times H \times W}$ |
| $y, B$ | Hedef (gerçek) görüntü | $\mathbb{R}^{3 \times H \times W}$ |
| $z$ | Gürültü vektörü | $\mathbb{R}^{d_z}$ |
| $p_{data}$ | Gerçek veri dağılımı | – |
| $p_g$ | Üretici dağılımı | – |
| $\mathcal{L}_{cGAN}$ | Koşullu çekişmeli kayıp | $\mathbb{R}^+$ |
| $\mathcal{L}_{L1}$ | L1 piksel kaybı | $\mathbb{R}^+$ |
| $\mathcal{L}_{cyc}$ | Döngü-tutarlılık kaybı (CycleGAN) | $\mathbb{R}^+$ |
| $\mathcal{L}_{VGG}$ | VGG algısal kaybı | $\mathbb{R}^+$ |
| $\lambda_{L1}$ | L1 kaybı ağırlık katsayısı | $\mathbb{R}^+$ (varsayılan 100) |
| $\lambda_{VGG}$ | VGG kaybı ağırlık katsayısı | $\mathbb{R}^+$ |
| $\alpha$ | Öğrenme oranı (learning rate) | $\mathbb{R}^+$ (varsayılan $2 \cdot 10^{-4}$) |
| $\beta_1, \beta_2$ | Adam optimizatörü momentum katsayıları | (0,5; 0,999) |
| $T$ | Toplam eğitim epoch sayısı | $\mathbb{N}$ |
| $T_{const}$ | Sabit LR fazı epoch sayısı | $\mathbb{N}$ |
| $\sigma$ | Gauss bulanıklaştırıcının standart sapması (XDoG) | $\mathbb{R}^+$ |
| $\gamma, \beta$ | SPADE'in uzamsal-uyarlamalı afin parametreleri | $\mathbb{R}^{C \times H \times W}$ |
| $\mu, \Sigma$ | Inception aktivasyonlarının ortalama ve kovaryansı (FID) | – |
| $\mathbf{m}$ | Anlamsal etiket maskesi (SPADE girdisi) | $\{0,\dots,K-1\}^{H \times W}$ |
| $K$ | Semantic sınıf sayısı | $\mathbb{N}$ |
| $\phi^l$ | Önceden eğitilmiş ağın $l$. katmanı aktivasyonu | – |

### Kısaltmalar

| Kısaltma | İngilizce Açılım | Türkçe Karşılık |
|---|---|---|
| **GAN** | Generative Adversarial Network | Üretken Çekişmeli Ağ |
| **cGAN** | Conditional GAN | Koşullu GAN |
| **DCGAN** | Deep Convolutional GAN | Derin Evrişimli GAN |
| **CycleGAN** | Cycle-Consistent GAN | Döngü-Tutarlı GAN |
| **SPADE** | Spatially-Adaptive (DE)normalization | Uzamsal-Uyarlamalı (De)normalizasyon |
| **CAIN-GAN** | Context-Aware Site Planning GAN | Bağlam-Farkındalıklı Saha Planlama GAN'ı |
| **CNN** | Convolutional Neural Network | Evrişimli Sinir Ağı |
| **U-Net** | U-shaped Encoder-Decoder Network | U-biçimli Encoder-Decoder Ağı |
| **VGG** | Visual Geometry Group (network) | VGG ağ ailesi |
| **PatchGAN** | Patch-based Discriminator | Yama-Tabanlı Ayırıcı |
| **LSGAN** | Least Squares GAN | En Küçük Kareler GAN |
| **WGAN-GP** | Wasserstein GAN with Gradient Penalty | Gradyan Cezalı Wasserstein GAN |
| **MSE** | Mean Squared Error | Ortalama Karesel Hata |
| **BCE** | Binary Cross-Entropy | İkili Çapraz Entropi |
| **L1** | L1 (Manhattan) distance | L1 Mesafesi |
| **FID** | Fréchet Inception Distance | Fréchet Inception Mesafesi |
| **SSIM** | Structural Similarity Index | Yapısal Benzerlik İndeksi |
| **PSNR** | Peak Signal-to-Noise Ratio | Tepe Sinyal-Gürültü Oranı |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Öğrenilmiş Algısal Görüntü Yaması Benzerliği |
| **XDoG** | eXtended Difference of Gaussians | Genişletilmiş Gauss Farkı |
| **DoG** | Difference of Gaussians | Gauss Farkı |
| **GPU** | Graphics Processing Unit | Grafik İşleme Birimi |
| **API** | Application Programming Interface | Uygulama Programlama Arayüzü |
| **GIS** | Geographic Information System | Coğrafi Bilgi Sistemi |
| **OSM** | OpenStreetMap | Açık Sokak Haritası |
| **LR** | Learning Rate | Öğrenme Oranı |
| **BAIR** | Berkeley Artificial Intelligence Research | Berkeley Yapay Zekâ Araştırma Merkezi |
| **CVPR** | IEEE Conf. on Computer Vision and Pattern Recognition | (Bilgisayarla Görü konferansı) |
| **ICCV** | IEEE Int. Conf. on Computer Vision | (Uluslararası Bilgisayarla Görü konferansı) |
| **ECCV** | European Conf. on Computer Vision | (Avrupa Bilgisayarla Görü konferansı) |
| **NeurIPS** | Conf. on Neural Information Processing Systems | (Sinir Ağ İşleme Sistemleri konferansı) |
| **ICLR** | Int. Conf. on Learning Representations | (Öğrenme Temsilcileri konferansı) |
| **UN DESA** | United Nations Department of Economic and Social Affairs | BM Ekonomik ve Sosyal İşler Departmanı |
| **WHO** | World Health Organization | Dünya Sağlık Örgütü |
| **MIT** | Massachusetts Institute of Technology (lisans) | MIT açık lisansı |
| **DOI** | Digital Object Identifier | Sayısal Nesne Tanımlayıcı |
| **CVF** | Computer Vision Foundation (open access) | – |
| **PMC** | PubMed Central | – |
| **PAT** | Personal Access Token | Kişisel Erişim Belirteci (GitHub) |

---

## TEZ YAZIM REHBERİ — DOSYA REFERANSI

> Bu tez şu dosyalardan derlenmiştir; FBE Word/LaTeX şablonuna kopyalama sırasında bu sıralama korunmalıdır.

```
docs/
├── 00_On_Sayfalar.md            ← BU DOSYA (özet, abstract, kısaltmalar)
├── 04_Bolum1_Giris.md           ← Bölüm 1
├── 03_Bolum2_Literatur.md       ← Bölüm 2
├── 05_Bolum3_Yontem.md          ← Bölüm 3
├── 06_Bolum4_Deneysel_Sonuclar.md  ← Bölüm 4 (yazılacak)
├── 07_Bolum5_Tartisma.md        ← Bölüm 5 (yazılacak)
├── 08_Bolum6_Sonuclar.md        ← Bölüm 6 (yazılacak)
├── 01_Arastirma_Raporu.md       ← (yazıma girmeyen) bağlam raporu
└── 02_Dataset_Karari.md         ← (yazıma girmeyen) karar günlüğü
```
