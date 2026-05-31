# 3. MATERYAL VE YÖNTEM

> **Bilgi notu:** Bu bölümde sunulan tüm algoritmalar, mimari yapılar ve hiperparametreler, açık kaynak olarak yayımlanan kod tabanında (https://github.com/ilker-23/gan-urban-design-v2) birebir uygulanmıştır. Her alt bölüm sonunda ilgili kaynak dosyasının repo içindeki yolu belirtilmektedir.

---

## 3.1 Genel Yaklaşım

Bu çalışmada, kentsel peyzaj plan üretimi ve renklendirme problemleri **koşullu üretken çekişmeli ağlar (cGAN)** [3], [4] paradigması altında **iki aşamalı bir görüntüden-görüntüye çeviri pipeline'ı** olarak formüle edilmiştir (Şekil 3.1):

```
                  ┌─────────────────────────────┐
   Aerial         │   Stage-1: Plan Üretimi     │   renk-kodlu
   görüntü   ────►│   (Pix2PixHD veya SPADE)    │──► semantic map
   (uydu)         │                             │
                  └─────────────────────────────┘
                                                          │
                  ┌─────────────────────────────┐         │
   Sentetik       │   Stage-2: Renklendirme     │   renkli   │
   kroki     ────►│   (Pix2Pix / CycleGAN /     │──► plan ◄──┘
   (Canny+XDoG)   │    Pix2PixHD / SPADE)       │
                  └─────────────────────────────┘
```

**Şekil 3.1.** Önerilen iki aşamalı kentsel peyzaj plan üretimi ve renklendirme pipeline'ı.

Her iki aşama da **eşli (paired) görüntü çiftleri** $(A, B) \in \mathcal{X} \times \mathcal{Y}$ üzerinde eğitilen bir cGAN ile gerçeklenmektedir. Burada $A$ koşul (girdi: uydu görüntüsü veya sentetik kroki), $B$ ise hedef (renk-kodlu plan) görüntüsüdür.

---

## 3.2 Veri Seti

### 3.2.1 Birincil Veri Seti: `maps` (Berkeley Pix2Pix)

Çalışmanın birincil dataset'i, Isola ve arkadaşlarının Pix2Pix çalışmasında [4] kullanıma sundukları açık erişimli `maps` koleksiyonudur. Veri seti **1096 eğitim + 1098 doğrulama = 2194 eşli görüntü çifti** içermektedir. Her çift, 1200×600 piksel boyutlarında tek bir görüntü dosyası olarak depolanmıştır; sol 600×600 piksel **uydu aerial görüntüsünü ($A_{sat}$)**, sağ 600×600 piksel ise **Google Maps render'ını ($B_{map}$)** temsil etmektedir.

Berkeley sunucusu üzerinden tek komutta indirilebilen veri seti, kayıt veya kimlik doğrulama gerektirmemektedir:

```bash
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
```

**Map tarafının renk kodlaması.** Google Maps render'ı standart bir renk paletine sahiptir: koyu yeşil = ağaçlık/park; açık yeşil = çim/açık alan; gri = yol/asfalt; beyaz = bina/yapay yüzey; mavi = su; sarı = ticari/yaya alanı. Bu renk kodlaması, kentsel peyzaj plan estetiğinin standart konvansiyonu ile birebir örtüşmekte ve bu nedenle dataset doğal olarak "renkli plan" hedefi rolünü üstlenmektedir.

> **Repo yolu:** `scripts/download_data.sh` (komut: `bash scripts/download_data.sh maps`).

### 3.2.2 İkincil Veri Seti: DeepGlobe Land Cover Classification

İkinci dataset olarak, Demir ve arkadaşlarının CVPR 2018 Workshop'unda tanıttığı **DeepGlobe Land Cover Classification** [23] kullanılmıştır. Bu veri seti **1.146 uydu görüntüsü × 2.448×2.448 piksel** çözünürlüğünde, **yedi sınıflı** ($K=7$) anlamsal segmentasyon maskeleri ile birlikte sunulmaktadır:

| Sınıf ID | Sınıf | Renk Kodu (RGB) |
|----------|-------|-----------------|
| 0 | urban (kentsel) | (0, 255, 255) |
| 1 | agriculture (tarım) | (255, 255, 0) |
| 2 | rangeland (otlak) | (255, 0, 255) |
| 3 | **forest (orman)** | (0, 255, 0) |
| 4 | water (su) | (0, 0, 255) |
| 5 | barren (çıplak) | (255, 255, 255) |
| 6 | unknown | (0, 0, 0) |

Bu veri seti **SPADE/GauGAN [7] eğitimi için Stage-1'in `semantic mask → aerial photo` çevirisi** alt-görevinde kullanılmaktadır.

> **Repo yolu:** `scripts/download_data.sh deepglobe` (Kaggle CLI ile).

### 3.2.3 Sentetik Kroki Üretim Algoritması

Bu çalışmada, **sketch → renkli plan** çevirisinin eğitimi için gerçek elle çizilmiş kroki yerine, mevcut renkli map görüntülerinden **sentetik B&W kroki** üretilmektedir. Bu yaklaşım, SketchyGAN [22] tarafından da kullanılan kabul görmüş bir veri-çoğaltma stratejisidir.

Sentetik kroki üretimi, **iki farklı edge detector'un birleşimi** olarak tasarlanmıştır:

**(a) Canny Edge Detector.** Standart Canny algoritması, görüntüdeki yüksek gradyan değişimlerini siyah-beyaz ikili çıktıya çevirir:

$$ E_{canny}(x,y) = \mathrm{Canny}\bigl(I_{gray}(x,y);\ T_{low}=50,\ T_{high}=150\bigr) \tag{3.1} $$

**(b) XDoG (eXtended Difference of Gaussians).** Yumuşak çizgi-sanatı estetiği için XDoG filtresi uygulanır:

$$ D(x,y) = G_{\sigma}(x,y) - \tau \cdot G_{k\sigma}(x,y) \tag{3.2} $$

$$ \mathrm{XDoG}(x,y) = \begin{cases} 1, & D(x,y) \geq \epsilon \\ 1 + \tanh(\varphi \cdot (D(x,y) - \epsilon)), & \mathrm{aksi} \end{cases} \tag{3.3} $$

Burada $G_{\sigma}$ ve $G_{k\sigma}$ farklı standart sapmalarda Gauss bulanıklaştırıcılar, $\tau \in (0,1)$ ağırlık katsayısı (0,98), $k=1{,}6$ sigma oranı, $\sigma=0{,}8$, $\epsilon=0{,}1$ eşik ve $\varphi=200$ keskinlik kontrol parametresidir.

**Birleşim:** Her iki edge map birleştirilerek nihai sentetik kroki elde edilir:

$$ S(x,y) = \min\bigl(\mathrm{XDoG}(x,y),\ 255 - E_{canny}(x,y)\bigr) \tag{3.4} $$

Bu birleşim, Canny'nin keskin kenarlarını XDoG'un yumuşak hatlarıyla zenginleştirir; elle çizilmiş kroki estetiğine **görsel olarak çok yakın** çıktılar üretir.

> **Repo yolu:** `scripts/prepare_sketches.py`, fonksiyon `make_sketch()`.

### 3.2.4 Veri Ön İşleme ve Augmentation

Eğitim için her görüntü çifti şu adımlardan geçirilir:

1. **Orta-ayırma:** 1200×600 birleşik görüntü, $A$ (sol 600²) ve $B$ (sağ 600²) olarak bölünür.
2. **Sentetik sketch üretimi:** $B$'den (3.4) eşitliği ile $S$ türetilir.
3. **Yeniden ölçeklendirme:** Bilineer interpolasyon ile 256×256 veya 512×512'ye küçültülür.
4. **Normalizasyon:** Piksel değerleri $[-1, 1]$ aralığına ölçeklenir: $x' = x/127{,}5 - 1$.
5. **Eğitim sırasında augmentation:**
   - Yatay flip (p = 0,5);
   - 0°/90°/180°/270° rastgele döndürme (p = 0,75);
   - Yalnızca $B$ tarafında hafif renk dalgalanması (p = 0,5, $\Delta \in [-15, +15]$ RGB).

Augmentation çiftin **paired** karakterini korur: $A$ ve $B$ aynı geometrik dönüşümden geçer.

> **Repo yolu:** `src/datasets/maps_sketch.py`, sınıf `PairedImageDataset`.

---

## 3.3 Önerilen GAN Mimarileri

Bu çalışmada dört farklı koşullu GAN mimarisi sistematik olarak karşılaştırılmaktadır. Her mimari, ortak değerlendirme protokolü altında aynı dataset üzerinde eğitilmiştir.

### 3.3.1 Pix2Pix (Baseline)

Pix2Pix [4], eşli görüntüden-görüntüye çeviri için **U-Net jeneratör + PatchGAN ayırıcı** mimarisini kullanır.

**Jeneratör — U-Net (Şekil 3.2).** U-Net, **simetrik encoder-decoder** yapısıyla ve **sıçramalı bağlantılarla (skip connections)** tasarlanmıştır. Bu çalışmada 256×256 girdi için **8 aşağı-örnekleyici (downsampling) bloğu** kullanılmıştır:

- Aşağı yön (encoder): Her blok 4×4 evrişim (stride=2) + InstanceNorm + LeakyReLU(0,2). Kanal sayısı 64'ten başlayıp her katmanda iki katına çıkar, maksimum 512'de doyar.
- Yukarı yön (decoder): Her blok 4×4 transpoze evrişim (stride=2) + InstanceNorm + ReLU. İlk üç decoder bloğunda Dropout(0,5) uygulanır.
- Sıçramalı bağlantılar: $i$. encoder bloğunun çıktısı, $(N-i)$. decoder bloğunun girdisine kanal-boyunca eklenir; bu sayede yüksek-frekans (kenar, kontur) bilgisi korunur.
- Son katman: Tanh aktivasyonu ile $[-1, 1]$ aralığında 3-kanallı çıktı.

**Ayırıcı — PatchGAN.** Ayırıcı, **70×70 piksel yamalara** gerçeklik kararı veren bir konvolüsyonel ağdır:

- Girdi: 6-kanallı yığın $\mathrm{concat}(A, B) \in \mathbb{R}^{6 \times H \times W}$.
- 4 katmanlı CNN; her katman 4×4 evrişim (stride=2 ilk üçte, stride=1 son ikide) + InstanceNorm + LeakyReLU.
- Çıktı: Yamasal gerçeklik haritası (logit; sigmoid uygulanmaz, LSGAN için MSE doğrudan hesaplanır).

PatchGAN, görüntünün **küresel anlamsal tutarlılığını L1 kaybı**na bırakırken **yüksek-frekans (keskin kenar, doku) gerçekliğini** ayırıcıya öğretir [4].

> **Repo yolu:** `src/models/networks.py` (sınıflar `UNetGenerator`, `PatchDiscriminator`) ve `src/models/pix2pix.py` (`Pix2PixModel`).

### 3.3.2 CycleGAN

CycleGAN [5], **eşsiz veri** için iki yönlü iki jeneratör kullanır: $G: X \to Y$ ve $F: Y \to X$. Karakteristik **döngü-tutarlılık (cycle-consistency)** kaybı (eşitlik 3.5):

$$ \mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_X}\bigl[\|F(G(x))-x\|_1\bigr] + \mathbb{E}_{y \sim p_Y}\bigl[\|G(F(y))-y\|_1\bigr] \tag{3.5} $$

Bu kayıp, içerik korumalı domain çevirisi sağlar ancak piksel-tam hassasiyeti Pix2Pix'in altındadır. Bu çalışmada CycleGAN, "eşli veri olmadığı varsayımı altında karşılaştırma baseline'ı" olarak konumlandırılmıştır.

> **Repo yolu:** Junyanz tarafından sunulan referans implementasyon kullanılmaktadır (`scripts/setup_external.sh` ile çekilir).

### 3.3.3 Pix2PixHD

Wang ve arkadaşları tarafından önerilen Pix2PixHD [6], Pix2Pix'i yüksek çözünürlüğe taşır. Üç ana yenilik:

**(i) Kaba-iniceye jeneratör (coarse-to-fine).** Jeneratör $G$ iki alt-ağdan oluşur: küçük çözünürlükte çalışan **global generator** $G_1$ ve onun çıktısını ince ayar yapan **local enhancer** $G_2$. Toplam jeneratör (eşitlik 3.6):

$$ G(\mathbf{x}) = G_2\bigl(\mathbf{x} \mathbin{\oplus} G_1(\mathrm{downsample}(\mathbf{x}))\bigr) \tag{3.6} $$

Burada $\oplus$ kanal-yönlü birleştirmeyi ifade eder.

**(ii) Çok-ölçekli ayırıcı.** Üç farklı ölçekte ($D_1$: 1×, $D_2$: 1/2×, $D_3$: 1/4×) PatchGAN ayırıcılar kullanılır:

$$ \mathcal{L}_{GAN}^{multi} = \sum_{i=1}^{3} \mathcal{L}_{cGAN}(G, D_i) \tag{3.7} $$

**(iii) Özellik eşleştirme + VGG algısal kaybı.** Eğitim kararlılığı için ayırıcının ara aktivasyonları arasında L1 mesafe minimize edilir, ek olarak önceden eğitilmiş VGG-19 ağında VGG-perceptual kaybı eklenir (eşitlik 3.8):

$$ \mathcal{L}_{VGG}(G) = \sum_{k} \lambda_k \cdot \|\phi_k(G(A)) - \phi_k(B)\|_1 \tag{3.8} $$

burada $\phi_k$, VGG-19'un $k$. konvolüsyon katmanının özellik haritasıdır.

> **Repo yolu:** NVIDIA'nın resmi implementasyonu (`scripts/setup_external.sh` ile çekilir). Bu tezde VGG-perceptual kaybı `src/losses.py` içinde `VGGPerceptualLoss` sınıfı olarak da bağımsız implement edilmiştir.

### 3.3.4 SPADE / GauGAN

Park ve arkadaşları tarafından önerilen SPADE [7], anlamsal görüntü sentezinde **uzamsal-uyarlamalı denormalizasyon** ile bir devrim yaratmıştır. Standart koşullu BatchNorm/InstanceNorm katmanlarının yerine SPADE katmanı kullanılır (eşitlik 3.9):

$$ \mathrm{SPADE}\bigl(h^i_{n,c,y,x} \mid \mathbf{m}\bigr) = \gamma^i_{c,y,x}(\mathbf{m}) \cdot \frac{h^i_{n,c,y,x} - \mu^i_c}{\sigma^i_c} + \beta^i_{c,y,x}(\mathbf{m}) \tag{3.9} $$

Burada:
- $h^i$: $i$. konvolüsyon katmanının aktivasyonu (boyut $N \times C \times H \times W$);
- $\mu^i_c$, $\sigma^i_c$: kanal-bazlı ortalama ve standart sapma (batch üzerinden);
- $\gamma^i_{c,y,x}(\mathbf{m})$, $\beta^i_{c,y,x}(\mathbf{m})$: **anlamsal maskeden öğrenilen** uzamsal-uyarlamalı afin parametreler.

Bu mekanizma, anlamsal etiket bilgisinin ağın derin katmanlarında "yıkanmasını" önler ve her piksel için **uzamsal olarak farklı bir afine dönüşüm** uygular. SPADE'in tezdeki rolü, **DeepGlobe semantic mask → uydu görüntüsü** çevirisinde Stage-1'in test edilmesidir.

> **Repo yolu:** NVlabs/SPADE resmi implementasyon (`scripts/setup_external.sh`).

---

## 3.4 Kayıp Fonksiyonları

### 3.4.1 Çekişmeli Kayıp Varyantları

Bu çalışmada üç farklı çekişmeli kayıp varyantı desteklenmektedir; konfigürasyon dosyasından (`configs/*.yaml`) seçilebilir:

**(a) Vanilla GAN (BCE).** Goodfellow'un orijinal log-olabilirlik formülasyonu [1]:

$$ \mathcal{L}_{vanilla} = \mathbb{E}\bigl[\log D(A, B)\bigr] + \mathbb{E}\bigl[\log(1 - D(A, G(A)))\bigr] \tag{3.10} $$

**(b) LSGAN (varsayılan).** Mao ve arkadaşları (2017) tarafından önerilen en-küçük-kareler GAN; sigmoid yerine doğrudan MSE kullanır; gradyan kaybolmasını azaltır:

$$ \mathcal{L}_{D}^{LS} = \tfrac{1}{2}\,\mathbb{E}\bigl[(D(A,B) - 1)^2\bigr] + \tfrac{1}{2}\,\mathbb{E}\bigl[D(A, G(A))^2\bigr] \tag{3.11} $$

$$ \mathcal{L}_{G}^{LS} = \tfrac{1}{2}\,\mathbb{E}\bigl[(D(A, G(A)) - 1)^2\bigr] \tag{3.12} $$

**(c) WGAN-GP.** Wasserstein mesafesi + gradyan cezası; en kararlı ama hesap-yoğun seçenek.

> **Repo yolu:** `src/losses.py`, sınıf `GANLoss` (parametre: `mode ∈ {"vanilla", "lsgan", "wgangp"}`).

### 3.4.2 L1 Piksel Kaybı

L1 mesafe, üretilen ile hedef arasında düşük-frekans (renk, genel düzen) tutarlılığını zorlar (eşitlik 3.13):

$$ \mathcal{L}_{L1}(G) = \mathbb{E}_{(A,B)}\bigl[\|B - G(A)\|_1\bigr] \tag{3.13} $$

Pix2Pix kayıp ağırlığı $\lambda_{L1} = 100$, deneylerimizde varsayılan değerdir.

### 3.4.3 VGG Algısal Kaybı (Pix2PixHD için)

Eşitlik (3.8)'de tanımlanmıştır. Bu tezde VGG-19'un $\{relu1\_2, relu2\_2, relu3\_3, relu4\_3\}$ katmanları kullanılır; sırasıyla ağırlıklar $\{\tfrac{1}{32}, \tfrac{1}{16}, \tfrac{1}{8}, \tfrac{1}{4}\}$.

### 3.4.4 Toplam Jeneratör Kaybı

Pix2Pix için toplam jeneratör kaybı (eşitlik 3.14):

$$ \mathcal{L}_G^{total} = \mathcal{L}_G^{LS} + \lambda_{L1} \cdot \mathcal{L}_{L1}(G) + \lambda_{VGG} \cdot \mathcal{L}_{VGG}(G) \tag{3.14} $$

Varsayılan: $\lambda_{L1} = 100$, $\lambda_{VGG} = 0$ (Pix2Pix baseline'da kapalı; Pix2PixHD'de 10).

---

## 3.5 Eğitim Protokolü

### 3.5.1 Optimizasyon

Tüm modeller **Adam optimizatörü** ile eğitilmiştir [4]:
- Öğrenme oranı: $\alpha = 2 \times 10^{-4}$;
- Momentum: $\beta_1 = 0{,}5$, $\beta_2 = 0{,}999$.

$\beta_1$'in düşük (0,5) seçimi Pix2Pix'in orijinal makalesinden gelen ampirik tavsiyedir ve GAN eğitiminde standart pratiktir.

### 3.5.2 Öğrenme Oranı Çizelgesi (LR Schedule)

İlk $T_{const}=100$ epoch sabit öğrenme oranıyla, kalan $T - T_{const}$ epoch boyunca **doğrusal azalan (linear decay)** öğrenme oranıyla eğitilir (eşitlik 3.15):

$$ \alpha(t) = \begin{cases} \alpha_0, & t \leq T_{const} \\ \alpha_0 \cdot \dfrac{T - t}{T - T_{const}}, & t > T_{const} \end{cases} \tag{3.15} $$

burada $T$ toplam epoch sayısı, $t$ mevcut epoch.

> **Repo yolu:** `src/utils.py`, fonksiyon `make_lambda_scheduler()`.

### 3.5.3 Diğer Hiperparametreler

| Hiperparametre | Değer | Kaynak |
|----------------|-------|--------|
| Görüntü boyutu | 256×256 (Pix2Pix), 512×512 (Pix2PixHD/SPADE) | [4], [6], [7] |
| Batch boyutu | 16 (256²) / 4 (512²) | A100 bellek kapasitesine göre |
| Toplam epoch sayısı $T$ | 200 (Pix2Pix), 150 (Pix2PixHD/SPADE) | [4], [6] |
| $T_{const}$ | 100 | [4] |
| $\lambda_{L1}$ | 100 | [4] |
| $\lambda_{VGG}$ | 0 (Pix2Pix), 10 (Pix2PixHD) | [6] |
| Generator base channels (ngf) | 64 | [4] |
| Discriminator base channels (ndf) | 64 | [4] |
| Norm tipi | InstanceNorm | [4] |
| Dropout (decoder ilk 3 katman) | 0,5 | [4] |
| Tekrarlanabilirlik için seed | 42 | – |

> **Repo yolu:** `configs/pix2pix_default.yaml`, `configs/pix2pix_sat2map.yaml`.

### 3.5.4 Eğitim Altyapısı

Tüm eğitimler **Google Colab Pro** ortamında, **NVIDIA A100 (40 GB)** GPU üzerinde yürütülmüştür. Her epoch sonunda checkpoint dosyaları Google Drive'a yazılmakta; bu, Colab oturumu kesilmesi durumunda eğitimin **resume edilebilirliğini** garanti eder.

Beklenen toplam GPU saati:

| Model | Çözünürlük | Epoch | A100 saat |
|-------|-----------|-------|-----------|
| Pix2Pix | 256² | 200 | ~1,0 |
| CycleGAN | 256² | 200 | ~2,0 |
| Pix2PixHD | 512² | 150 | ~4,5 |
| SPADE | 512² | 150 | ~6,0 |
| **Toplam** | | | **~13,5** |

> **Repo yolu:** `notebooks/colab_pix2pix.ipynb`.

---

## 3.6 Değerlendirme Metrikleri

Üretilen çıktıların kalitesi, beş tamamlayıcı metrik ile değerlendirilmektedir. Her metrik val seti üzerinde örnek-bazında hesaplanır; sonuçlar ortalama ve standart sapma olarak raporlanır.

### 3.6.1 Fréchet Inception Distance (FID)

Heusel ve arkadaşları (2017) tarafından önerilen FID [24], gerçek ve sentetik dağılımları önceden-eğitilmiş Inception-V3 ağının özellik uzayında karşılaştırır (eşitlik 3.16):

$$ \mathrm{FID}(P_r, P_g) = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\bigl(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\bigr) \tag{3.16} $$

burada $\mu$ ve $\Sigma$, Inception-V3 son havuzlama (pool3) katmanı aktivasyonlarının (2048 boyutlu) ortalama ve kovaryans matrisidir. **Düşük FID değeri daha iyidir.**

### 3.6.2 Structural Similarity Index (SSIM)

Wang ve arkadaşları (2004) tarafından önerilen SSIM, görüntüleri **luminans + kontrast + yapı** üç bileşeniyle karşılaştırır (eşitlik 3.17):

$$ \mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)} \tag{3.17} $$

burada $c_1 = (0{,}01 \cdot L)^2$, $c_2 = (0{,}03 \cdot L)^2$ ve $L = 255$ piksel dinamik aralığı. SSIM değeri $[-1, 1]$ aralığında olup **1'e yakın değer daha iyidir.**

### 3.6.3 Peak Signal-to-Noise Ratio (PSNR)

Klasik piksel-tabanlı kalite metriği (eşitlik 3.18):

$$ \mathrm{PSNR}(x, y) = 10 \log_{10} \frac{L^2}{\mathrm{MSE}(x, y)} \tag{3.18} $$

burada $\mathrm{MSE}(x, y) = \tfrac{1}{HW}\sum_{i,j}(x_{ij}-y_{ij})^2$. **Yüksek PSNR daha iyidir** (genellikle 25-40 dB arası kabul edilir).

### 3.6.4 Learned Perceptual Image Patch Similarity (LPIPS)

Zhang ve arkadaşları (2018) tarafından önerilen LPIPS, derin CNN özellik aktivasyonlarında öğrenilen algısal uzaklıktır. Bu çalışmada AlexNet backbone kullanılmıştır:

$$ \mathrm{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \|w_l \odot (\hat{\phi}^l_{hw}(x) - \hat{\phi}^l_{hw}(y))\|_2^2 \tag{3.19} $$

burada $\hat{\phi}^l$ normalize edilmiş AlexNet aktivasyonları, $w_l$ öğrenilen kanal-bazlı ağırlıklar. **Düşük LPIPS daha iyidir.**

### 3.6.5 L1 Ortalama Mutlak Hata

Tamamlayıcı piksel-bazlı metrik (eşitlik 3.20):

$$ \mathrm{L1}(x, y) = \frac{1}{HW C}\sum_{i,j,c} |x_{ijc} - y_{ijc}| \tag{3.20} $$

**Düşük L1 daha iyidir.**

> **Repo yolu:** `src/evaluate.py`, fonksiyonlar `compute_pixel_metrics()`, `compute_lpips()`, `compute_fid()`.

---

## 3.7 Yazılım Mimarisi ve Tekrarlanabilirlik

### 3.7.1 Repository Yapısı

Çalışma, MIT lisansı altında halka açık olan https://github.com/ilker-23/gan-urban-design-v2 deposunda yayımlanmıştır. Klasör organizasyonu:

```
gan-urban-design-v2/
├── configs/        # YAML eğitim konfigürasyonları
├── data/           # Veri seti dizini (gitignore — Drive/Colab'da)
├── docs/           # Araştırma raporu, dataset kararı, tez bölümleri
├── notebooks/      # Colab notebook'ları
├── scripts/        # İndirme, sentetik sketch, external repo setup
└── src/
    ├── datasets/   # PairedImageDataset
    ├── models/     # UNetGenerator, PatchDiscriminator, Pix2PixModel
    ├── losses.py   # GANLoss, VGGPerceptualLoss
    ├── train.py    # Eğitim girdisi
    ├── evaluate.py # FID/SSIM/PSNR/LPIPS/L1
    └── utils.py    # seed, config, scheduler
```

### 3.7.2 Tekrarlanabilirlik

Aşağıdaki üç adım, sonuçların tam tekrarlanabilirliğini sağlar:

1. **Sabit rastgelelik tohumu:** Python random, NumPy, PyTorch (CPU+GPU) için seed=42 (`src/utils.py::set_seed`).
2. **Versiyonlanmış bağımlılıklar:** `requirements.txt` minimum versiyonları sabitler (PyTorch ≥ 2.1, vb.).
3. **Konfigürasyon serileştirmesi:** Eğitim başlangıcında çözümlenmiş YAML konfigürasyonu `${RUN_DIR}/config_resolved.json` olarak yazılır.

### 3.7.3 Tek Komutta Yeniden Üretim

Aşağıdaki komutlar zinciri, sıfırdan tüm pipeline'ı çalıştırır:

```bash
git clone https://github.com/ilker-23/gan-urban-design-v2.git
cd gan-urban-design-v2
pip install -r requirements.txt
bash scripts/download_data.sh maps
python scripts/prepare_sketches.py
python -m src.train --config configs/pix2pix_default.yaml
python -m src.evaluate --config configs/pix2pix_default.yaml \
                       --checkpoint results/pix2pix_default/checkpoints/pix2pix_epoch_0200.pth
```

---

## 3.8 Bölümün Özeti

Bu bölümde:

- **Veri seti** olarak halka açık `maps` (Berkeley, 2194 çift) ve `DeepGlobe Land Cover` (1.146 görüntü, 7-sınıflı mask) tanıtıldı.
- **Sentetik kroki üretimi** için Canny + XDoG birleşim algoritması formel olarak verildi.
- Karşılaştırılan **dört GAN mimarisi** (Pix2Pix, CycleGAN, Pix2PixHD, SPADE) matematiksel formülasyonlarıyla sunuldu.
- **Üç çekişmeli kayıp varyantı** (Vanilla, LSGAN, WGAN-GP), **L1 piksel kaybı** ve **VGG algısal kaybı** açıklandı.
- **Eğitim hiperparametreleri** ve **Colab Pro A100 altyapısı** tanımlandı.
- **Beş değerlendirme metriği** (FID, SSIM, PSNR, LPIPS, L1) formel olarak verildi.
- Tezin **açık kaynak repository yapısı** ve **tekrarlanabilirlik garantileri** belgelendi.

Bölüm 4'te bu metodolojinin deneysel sonuçları sunulmaktadır.
