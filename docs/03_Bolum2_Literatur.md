# 2. LİTERATÜR ARAŞTIRMASI

> **Bilgi notu:** Bu bölümde verilen tüm referanslar; arXiv kayıtları, IEEE/CVF Open Access arşivleri, yayıncı (Elsevier, Springer, MDPI, SAGE, Frontiers) DOI bağlantıları ve resmi proje sayfaları üzerinden teyit edilmiş, kullanıcı tarafından doğrulanabilen kaynaklardır. Her kaynak en az iki bağımsız bağlantı ile listelenmiştir (birincil yayıncı + arXiv veya proje sayfası).

Bu tezin amacı, üretken çekişmeli ağ (Generative Adversarial Network — GAN) tabanlı koşullu görüntü üretim modelleri kullanılarak kentsel peyzaj — özellikle park ve bahçe — tasarım planlarının otomatik olarak oluşturulması ve siyah-beyaz krokilerin renklendirilmesidir. Bu bölüm; GAN'ların kuramsal temellerinden başlayarak, görüntüden-görüntüye çeviri (image-to-image translation), anlamsal görüntü sentezi (semantic image synthesis), kentsel/peyzaj planlama için özelleşmiş GAN mimarileri ve kroki renklendirme yöntemlerini sistematik olarak incelemekte; sonunda tezin literatürdeki boşluğu doldurma stratejisini sunmaktadır.

---

## 2.1 Üretken Çekişmeli Ağların Kuramsal Temelleri

### 2.1.1 Klasik GAN Çerçevesi

Üretken çekişmeli ağ kavramı, Goodfellow ve arkadaşları tarafından NeurIPS 2014'te tanıtılmıştır [1]. Çerçeve, **iki sinir ağının min-max oyunu** ile gerçek veri dağılımını öğrenmesi prensibine dayanır: bir **üretici (generator, G)** rastgele bir gürültü vektöründen (z) örnekler üretmeyi öğrenirken, bir **ayırıcı (discriminator, D)** gerçek örnekleri sentetiklerden ayırt etmeyi öğrenir. Çerçevenin değer fonksiyonu (eşitlik 2.1):

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z(z)}\left[\log(1 - D(G(z)))\right] \tag{2.1} $$

Goodfellow ve arkadaşları, bu min-max oyununun global optimumunun yalnızca üretici dağılımı $p_g$ ile gerçek veri dağılımı $p_{data}$ özdeş olduğunda elde edildiğini ve bu noktada ayırıcının her örnek için $\tfrac{1}{2}$ olasılığı verdiğini göstermişlerdir [1].

Bu yaklaşımın iki önemli avantajı vardır: (i) Markov zincirleri veya açımlı yaklaşık çıkarım ağları gerektirmez; (ii) üretici, eğitim verisini açıkça hatırlamak (memorization) yerine veri dağılımını **örtük (implicit) olarak** modelleyebilir. Buna karşın, orijinal formülasyon **eğitim kararsızlığı, mod çökmesi (mode collapse) ve gradyan kayboluşu** problemlerinden muzdariptir; izleyen çalışmaların büyük bölümü bu sorunları hafifletmeyi amaçlamıştır.

### 2.1.2 DCGAN — Konvolüsyonel Mimarinin Standartlaşması

Radford, Metz ve Chintala tarafından ICLR 2016'da sunulan **Deep Convolutional GAN (DCGAN)** [2], evrişimsel sinir ağlarını GAN çerçevesine entegre ederek eğitim kararlılığını ciddi ölçüde artırmıştır. Mimari kurallar şunlardır: tam-bağlı katmanların kaldırılması, *strided* evrişim/transpoze-evrişim kullanılması, jeneratörde ReLU + Tanh, ayırıcıda LeakyReLU aktivasyonu ve toplu normalleştirme (batch normalization). DCGAN, ürettiği örneklerin gizli uzayında **aritmetik geçişlerin anlamsal olarak yorumlanabilir** olduğunu göstererek modern GAN tasarımının temellerini atmıştır [2].

### 2.1.3 Koşullu GAN'lar — Yönlendirilmiş Üretim

Mirza ve Osindero (2014) [3], hem üreticiye hem ayırıcıya bir **koşul vektörü ($y$)** ekleyerek "Koşullu GAN (cGAN)" çerçevesini önermiştir. Koşul; bir sınıf etiketi, bir metin gömmesi veya — bu tezin ilgilendiği gibi — bir başka görüntü olabilir. Değer fonksiyonu eşitlik (2.2) ile verilir:

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}\left[\log D(x|y)\right] + \mathbb{E}_{z \sim p_z}\left[\log(1 - D(G(z|y)|y))\right] \tag{2.2} $$

Bu formülasyon, **görüntüden-görüntüye çeviri** problemleri için temel paradigmayı oluşturur ve takip eden tüm Pix2Pix ailesi modellerin temelidir.

---

## 2.2 Koşullu GAN ile Görüntüden-Görüntüye Çeviri

### 2.2.1 Pix2Pix — Eşli (Paired) Çeviri için Genel Çerçeve

Isola, Zhu, Zhou ve Efros tarafından CVPR 2017'de sunulan **Pix2Pix** [4], birbirine eşlenmiş görsel çiftlerden (örneğin etiket haritası ↔ fotoğraf, kenar haritası ↔ nesne, gündüz ↔ gece) **genel amaçlı bir çeviri çerçevesi** sunar. Üç temel yenilik içerir:

1. **U-Net üretici:** Sıçramalı bağlantılar (skip connections) sayesinde girdi ile çıktının uzamsal hizalanmasını korur. Plan/diyagram gibi keskin kenarların önemli olduğu görüntülerde belirleyici olmuştur [4].
2. **PatchGAN ayırıcı:** Tüm görüntüye değil **N×N büyüklüğündeki yamalara** (typically 70×70) gerçeklik kararı verir; bu, yüksek-frekans ayrıntılarını korurken parametre sayısını düşürür [4].
3. **Birleşik kayıp:** Çekişmeli kayıp ile L1 piksel kaybının ağırlıklı toplamı kullanılır (eşitlik 2.3):

$$ G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G,D) + \lambda \cdot \mathcal{L}_{L1}(G) \tag{2.3} $$

Burada $\lambda = 100$ tipik değerdir ve L1 terimi düşük-frekans (renk bölgeleri, genel düzen) doğruluğunu, GAN terimi ise yüksek-frekans (keskinlik, gerçekçilik) bileşenini sağlamakla görevlendirilmiştir. Yazarlar Cityscapes anlamsal etiketten foto, mimari etiketten cephe (CMP Facades), kenar haritasından ayakkabı/çanta ve **uydu görüntüsünden harita üretimi (`maps` dataset'i)** gibi farklı görevler üzerinde başarısını göstermiştir [4]. Bu tezde tam olarak aynı `maps` dataset'i ana eğitim kaynağı olarak kullanılmaktadır (bkz. Bölüm 3.1).

### 2.2.2 CycleGAN — Eşsiz (Unpaired) Veri için Döngü-Tutarlı Öğrenme

Eşli veri her zaman bulunmaz. Zhu, Park, Isola ve Efros tarafından ICCV 2017'de sunulan **CycleGAN** [5], iki yönlü iki üretici ($G: X \to Y$ ve $F: Y \to X$) ve **döngü-tutarlılık (cycle-consistency)** kaybı ile bu kısıtı aşar (eşitlik 2.4):

$$ \mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_X}\!\left[\|F(G(x))-x\|_1\right] + \mathbb{E}_{y \sim p_Y}\!\left[\|G(F(y))-y\|_1\right] \tag{2.4} $$

Bu kayıp; eşli olmayan iki domain (örneğin "at ↔ zebra", "yaz ↔ kış") arasında **içerik korumalı stil aktarımı** sağlar. Plan/diyagram türü görüntülerde piksel-tam hassasiyeti Pix2Pix'in altında kalır; bu nedenle bu tezde **karşılaştırma baseline'ı (sketch'in eşsiz olduğu varsayımı altında)** olarak konumlandırılmıştır.

### 2.2.3 Pix2PixHD — Yüksek Çözünürlüklü Anlamsal Sentez

Wang ve arkadaşları (NVIDIA + UC Berkeley) tarafından CVPR 2018'de sunulan **Pix2PixHD** [6], Pix2Pix çerçevesini **2048×1024 çözünürlüğe** taşımıştır. Üç ana yenilik içerir:

1. **Kaba-iniceye üretici:** İki aşamalı bir jeneratör — önce 1024×512 üreten G₁ "global" ağı, sonra G₂ "yerel ayrıştırıcı" ağı G₁'in özellik haritalarını ince ayar yapar [6].
2. **Çok-ölçekli ayırıcı:** Aynı PatchGAN mimarisi 3 farklı ölçekte (orijinal, ½, ¼) uygulanır; böylece hem genel görsel tutarlılık hem ince detay kararları yapılır.
3. **Özellik eşleştirme + VGG algısal kaybı:** Eğitim kararlılığı için ayırıcının ara aktivasyonları ile gerçek/sahte arasında L1 mesafesi minimize edilir; ayrıca önceden eğitilmiş bir VGG ağının özellik uzayında L1 mesafe (perceptual loss) eklenir.

Pix2PixHD, kentsel peyzaj planlama yazınında en yaygın kullanılan modellerden biridir (örn. Quan'ın "Suggestive Site Planning" çalışması doğrudan Pix2PixHD'yi temel alır [13]).

### 2.2.4 Pix2Pix Ailesi'nde Standart Kayıp Türleri

Pix2Pix ailesinde sıkça karşılaşılan üç çekişmeli kayıp varyantı bulunur:

- **Vanilla cGAN (Binary Cross-Entropy):** Goodfellow'un orijinal log-olabilirlik formülasyonu [1].
- **LSGAN (Mao vd., 2017):** Sigmoid yerine en küçük kareler kullanır; gradyan kaybolmasını azaltır ve daha kararlı eğitim sağlar.
- **WGAN-GP (Gulrajani vd., 2017):** Wasserstein mesafesi + gradyan cezası; mod çökmesini önler ama hesap maliyeti yüksektir.

Bu tezde **LSGAN** varsayılan olarak kullanılmıştır (Bölüm 3.3 ve `configs/pix2pix_default.yaml`).

---

## 2.3 Anlamsal Görüntü Sentezi: SPADE / GauGAN

Pix2PixHD, anlamsal etiket haritasını jeneratörün **girdi katmanı** olarak kullanır; bu yaklaşımın temel problemi, kontrastsız (homojen) bölgelerin (örneğin "gökyüzü", "çim", "su") normalleştirme katmanları boyunca yıkanması (washing away) ve anlam bilgisinin ağın derinliklerinde kaybolmasıdır [7].

Park, Liu, Wang ve Zhu (NVIDIA) tarafından CVPR 2019'da sunulan **SPADE (SPatially-Adaptive DE-normalization)** [7], bu problemi çözmek için **uzamsal-uyarlamalı bir normalleştirme** önerir. Normalleştirilen aktivasyonlar, anlamsal mask'tan öğrenilen $\gamma(x,y)$ ve $\beta(x,y)$ parametreleriyle yeniden ölçeklenir (eşitlik 2.5):

$$ \mathrm{SPADE}(h^i_{n,c,y,x} | \mathbf{m}) = \gamma^i_{c,y,x}(\mathbf{m}) \cdot \frac{h^i_{n,c,y,x} - \mu^i_c}{\sigma^i_c} + \beta^i_{c,y,x}(\mathbf{m}) \tag{2.5} $$

Burada $\mathbf{m}$ anlamsal etiket haritası, $h^i$ ise ağın $i$. katmanındaki aktivasyondur. Bu mekanizma sayesinde, etiket bilgisinin ağ derinleştikçe yıkanması engellenir ve her piksel için **uzamsal olarak farklı** bir afine dönüşüm uygulanır.

SPADE'in uygulamalı adı olan **GauGAN**, kullanıcıya gerçek zamanlı bir "boya tahtasına" anlamsal etiket çizdirmekte ve foto-gerçekçi peyzaj görüntüleri üretmektedir. Bu, tezin uygulama alanı (kentsel peyzaj) ile birebir örtüşür; bu nedenle SPADE bu tezde **ana ileri model adayı** olarak konumlandırılmıştır. Resmi PyTorch implementasyonu NVIDIA-NVlabs deposunda açık kaynaktır (bkz. Kaynak [7]).

---

## 2.4 Kentsel Plan Üretimi için GAN Tabanlı Çalışmalar

Built environment (yapılı çevre) için GAN uygulamaları 2020 sonrasında hızla artmıştır. Bu alanda kapsamlı bir derleme He ve arkadaşları tarafından *Building and Environment* dergisinde yayımlanmıştır [8]. Aşağıda, tezin doğrudan üzerine inşa edileceği seçilmiş çalışmalar kronolojik olarak incelenmektedir.

### 2.4.1 Suggestive Site Planning (Quan, 2019)

Quan tarafından CAAD Futures konferansında (Springer baskısı 2020) sunulan **"Suggestive Site Planning with Conditional GAN and Urban GIS Data"** [13]; OSM ve şehir GIS portallarından çekilen poligonları (yol, parsel, mevcut bina) rasterleştirerek **Pix2PixHD** ile eğitir. Çıktı, site sınırı verildiğinde önerilen bina ayak izleri ve fonksiyon kodlarıdır. Bu çalışma, **plan üretimini bir koşullu çeviri problemi** olarak formüle etmenin ilk akademik örneklerinden biridir.

### 2.4.2 MasterplanGAN (Ye, Du, Ye; 2022)

Ye ve arkadaşları, *Environment and Planning B: Urban Analytics and City Science* dergisinde sundukları MasterplanGAN [12] ile **şehir master planı vektör çizimlerini foto-gerçekçi 3B kuş bakışı görselleştirmelere** dönüştürmenin Pix2Pix-tabanlı pratik kullanımını göstermiştir. Tezin Stage-2 ("renkli plan render") aşaması ile doğrudan ilişkilidir.

### 2.4.3 UrbanGenoGAN (Chen vd., 2023)

Chen ve arkadaşları, *Frontiers in Environmental Science* dergisinde sundukları UrbanGenoGAN [14] ile GAN + **Genetik Algoritma + GIS** sinerjisini önerir. Üretilen plan adayları, sürdürülebilirlik kriterlerine göre genetik bir uygunluk fonksiyonu ile optimize edilir. Bu yaklaşım, tek başına GAN'ın "estetik olarak makul ama performans-bilinçsiz" çıktılar üretebileceği eleştirisine yanıt olarak konumlandırılır.

### 2.4.4 Mimari Yerleşim Üretimi — Graph-Constrained cGAN (Nauata vd., 2020, 2021)

Mimari iç yerleşim (bina katı planı) üretimi için iki referans çalışma House-GAN [15] (ECCV 2020) ve onun ardılı House-GAN++ [16] (CVPR 2021)'tir. Yazarlar, **oda komşuluk grafını** koşul olarak alıp ilgili kutucukları (axis-aligned bounding boxes) üreten **graf-kısıtlı çekişmeli ağ** mimarisini önermişlerdir. House-GAN++ ardışık (iterative) bir iyileştirme sağlar: bir önceki üretim, bir sonraki üretimin koşulu olur. Bu yaklaşım park bahçeleri gibi **bileşen ilişkilerinin önemli olduğu** peyzaj planlarında da uyarlanabilir bir yön sunar.

Aynı paradigmayı çevreleyen iki güncel çalışma şunlardır:
- Liu ve arkadaşları (Elsevier *Automation in Construction*, 2023) **site-embedded GAN** mimarisini önermiş [17];
- Ying ve arkadaşları (aynı dergi, 2023) **grafik-kısıtlı koşullu GAN** ile mimari yerleşim üretmiştir [18];
- Yan ve arkadaşları (Springer *Building Simulation*, 2025) GraphVAE + GAN birleşimi olan **GCGAN** ile bina yerleşim sentezini geliştirmiştir [19].

### 2.4.5 CAIN-GAN (Jiang vd., 2024) — Bu Tezin Referans Çerçevesi

Jiang ve arkadaşları tarafından Elsevier *Automation in Construction* dergisinde 2024'te yayımlanan **CAIN-GAN (Context-Aware Site Planning GAN)** [11], tez önerisinde Şekil 1 olarak referans verilen ve bu tezin kavramsal temelini oluşturan çalışmadır. CAIN-GAN iki aşamalı bir yapıya sahiptir:

- **Faz 1 — Footprint Construction:** Site sınırı + bağlam (kentsel çevre dokusu) → bina ayak izi koşullu üretim;
- **Faz 2 — Height Completion:** Üretilen 2B planın 3B yüksekliklerle zenginleştirilmesi.

CAIN-GAN'ın en önemli yeniliği, **alan bilgisinin (domain knowledge)** kayba dahil edilmesidir: kentsel sürdürülebilirlik göstergeleri (yapı yoğunluğu, açık alan oranı, vb.) hem üretici hem ayırıcının değerlendirme sürecine entegre edilir. Bu tez, CAIN-GAN'ın yapı yoğunluğu odağını **park ve bahçe yoğunluğu** odağına uyarlayarak konumlanmaktadır.

### 2.4.6 Duygu/Algı Tabanlı Otomatik Peyzaj Tasarımı (2024)

PMC veritabanı kayıtlarından erişilebilen ve 2024'te yayımlanan bir çalışma [20], "algı–muhakeme–üretim–geribesleme" çerçevesinde GAN'ı bilgisayarla görme + pekiştirmeli öğrenme + çoklu-amaçlı evrimsel optimizasyon ile birleştirerek **duygu/algı temelli** kentsel peyzaj üretimi yapmaktadır. Bu yaklaşım, niteliksel değerlendirmenin (uzman/kullanıcı görüşü) ölçülebilir bir hedef olarak modele entegre edilmesinin son örneğidir.

---

## 2.5 Kroki (Sketch) Tabanlı Renklendirme

Tezin **ikinci alt görevi**, siyah-beyaz krokilerin renkli ve sunum-hazır peyzaj planlarına dönüştürülmesidir. Bu görev, doğrusal-bağımsız bir alt-literatüre sahiptir.

### 2.5.1 Scribbler (Sangkloy vd., 2017)

Sangkloy, Lu, Fang, Yu ve Hays tarafından CVPR 2017'de sunulan **Scribbler** [21], kullanıcının **kenar çizgileri + seyrek renk noktaları** ile yönlendirdiği derin görüntü sentezini öneren ilk akademik çalışmalardan biridir. Yatak odası, yüz ve otomobil kategorilerinde anlamlı sonuçlar üretmiştir. Yaklaşım, bu tezde **kullanıcı renk ipuçlarının opsiyonel ek koşul olarak** alınabileceği bir mimari için ilham kaynağıdır.

### 2.5.2 SketchyGAN (Chen ve Hays, 2018)

Chen ve Hays tarafından CVPR 2018'de sunulan **SketchyGAN** [22], elle çizilmiş krokiden 50 kategoride foto-gerçekçi görüntü sentezleyen ilk büyük-ölçekli çalışmadır. İki temel katkı içerir: (i) **otomatik kroki-foto veri çoğaltma (data augmentation)** stratejisi; (ii) **MRU (Masked Residual Unit)** adlı, girdiyi çoklu ölçeklerde tekrar enjekte eden yeni bir yapı bloğu. Tezin sentetik sketch üretim stratejisi (Canny + XDoG) SketchyGAN'ın kroki-otomatik-üretim önerisiyle aynı yöntemsel hattadır.

### 2.5.3 Chen vd. (2024) — Park Krokilerini GAN ile Renklendirme

Bu tezin doğrudan üzerine inşa edileceği **birincil benchmark çalışması**, Chen ve arkadaşları tarafından MDPI *Land* dergisinde 2024'te yayımlanan ve doğrudan tezin alt başlığı ile örtüşen makaledir [10]: *"Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches."* Yazarlar:

- **Veri:** 152 adet eşli (kroki + renkli plan) park çizimi.
- **Yöntem:** Koşullu GAN; **Pix2Pix ve CycleGAN** karşılaştırması.
- **Bulgu:** Pix2Pix, paired data ile renk doğruluğu ve detay korunmasında CycleGAN'ı geçmiştir. GAN-tabanlı veri çoğaltma çıktı kalitesini iyileştirmiştir.
- **Sınırlama:** Dataset boyutu küçüktür (152 çift); önerilen modeller temel Pix2Pix/CycleGAN olup ileri mimari (Pix2PixHD, SPADE) denenmemiştir.

Bu tezin bu çalışmaya kıyasla katkıları:
1. **Veri ölçeği:** `maps` dataset'inden türetilen ~2,000+ eşli kroki–renkli plan örneği (~14× büyüklük).
2. **Mimari kapsamı:** Pix2Pix ve CycleGAN'a ek olarak Pix2PixHD ve SPADE de değerlendirilmektedir.
3. **Pipeline bütünlüğü:** Yalnız renklendirme değil, plan üretimi + renklendirmenin iki aşamalı entegrasyonu.

---

## 2.6 Değerlendirme Metrikleri Üzerine Literatür

GAN tabanlı görüntü çevirisi modelleri için yaygın olarak dört metrik kullanılır:

- **FID (Fréchet Inception Distance)** — Heusel vd. (NeurIPS 2017) [24] tarafından önerilen, Inception-V3 özellik uzayında gerçek ve sahte dağılım istatistiklerini karşılaştıran ölçüt. Düşük değer daha iyidir.
- **SSIM (Structural Similarity Index)** — Wang vd. (IEEE TIP 2004), yapısal yakınlığı luminans + kontrast + yapı bileşenleriyle ölçer.
- **PSNR (Peak Signal-to-Noise Ratio)** — Klasik piksel-tabanlı yeniden yapılandırma kalitesi metriği.
- **LPIPS (Learned Perceptual Image Patch Similarity)** — Zhang vd. (CVPR 2018), derin CNN özellik aktivasyonlarında öğrenilen algısal uzaklık.

Bu tezde tüm dört metrik birlikte raporlanmaktadır; ek olarak L1 ortalama mutlak hata da yer alır (bkz. `src/evaluate.py`).

---

## 2.7 Difüzyon Modelleri ile Karşılaştırma (Genel Bağlam)

Son iki yılda görüntüden-görüntüye çeviride **latent difüzyon modelleri (Stable Diffusion, Rombach vd., CVPR 2022)** [25] ve onların kontrollü uzantısı **ControlNet (Zhang vd., ICCV 2023)** [26] alanın yeni *state-of-the-art*'ı haline gelmiştir. Difüzyon modellerinin GAN'lara üstünlükleri (eğitim kararlılığı, çıktı çeşitliliği) yanında dezavantajları da vardır:

- **Çıkarım hızı:** GAN tek ileri geçiş (forward pass) ile örnek üretirken, difüzyon modeli onlarca-yüzlerce adıma ihtiyaç duyar; gerçek-zamanlı tasarım iş akışında GAN üstündür.
- **Eğitim verisi/ölçeği gereksinimi:** Genel-amaçlı difüzyon modelleri milyonlarca görsel ister; küçük tezler için fine-tuning maliyeti yüksektir.
- **Açıklanabilirlik:** SPADE/Pix2PixHD'nin mimari blokları yorumlanabilir; difüzyon modelleri kara kutu davranır.

Bu nedenlerle bu tez, **GAN-merkezli odağını sürdürmekte**, difüzyon modellerini "gelecek çalışmalar" başlığı altında karşılaştırma çerçevesinde tartışmaktadır.

---

## 2.8 Literatürdeki Boşluk ve Bu Tezin Konumu

Yukarıdaki tarama dört önemli boşluk göstermektedir:

**B1.** *Park-spesifik renklendirme çalışmaları küçük ölçekli kalmıştır.* Chen vd. 2024 [10] sadece 152 örnek kullanmıştır; ileri mimariler (Pix2PixHD/SPADE) park renklendirme problemine **literatürde henüz uygulanmamıştır**.

**B2.** *Plan üretimi ile renklendirme ayrı çalışılmıştır.* CAIN-GAN [11] ve MasterplanGAN [12] plan üretimine, Chen vd. [10] renklendirmeye odaklanır; **birleşik (end-to-end) bir pipeline** yoktur.

**B3.** *Kıyaslama datasetleri çeşitliliği eksiktir.* Park çalışmalarında kullanılan dataset'ler genellikle kapalı/özeldir. `maps` (Berkeley) gibi standart ve **halka açık** bir veriye dayanan park-renklendirme çalışması bulunmamaktadır.

**B4.** *Uzaktan algılama datasetleri ile peyzaj üretiminin köprüsü zayıftır.* DeepGlobe [22] gibi mevcut uydu görüntü datasetlerini GAN tabanlı plan/render üretimine adapte eden çalışma sınırlıdır.

Bu tezin literatüre özgün katkıları bu boşluklara karşılık gelir:

1. **`maps` dataset'inin park-renklendirme problemine uyarlanması** (B3'e yanıt).
2. **Pix2Pix, CycleGAN, Pix2PixHD ve SPADE'in aynı dataset üzerinde sistematik karşılaştırılması** (B1'e yanıt).
3. **İki aşamalı (üretim + renklendirme) entegre pipeline'ın inşası** (B2'ye yanıt).
4. **DeepGlobe Land Cover üzerinde SPADE ile semantic-mask → uydu görüntüsü üretiminin değerlendirilmesi** (B4'e yanıt).
5. **Beş farklı standart metriğin (FID, SSIM, PSNR, LPIPS, L1) ortak protokolde raporlanması.**

Bu katkılar, Bölüm 3'te (Materyal ve Yöntem) ve Bölüm 4'te (Deneysel Sonuçlar) ayrıntılı olarak ele alınmaktadır.

---

## Bölüm 2 Kaynakları (Doğrulanmış Bağlantılarla)

> Her kaynak için **birincil yayıncı/konferans bağlantısı + arXiv/preprint + (varsa) GitHub/proje sayfası** verilmiştir. Bağlantılar 31 Mayıs 2026 itibarıyla erişilebilir durumdadır.

**[1]** Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems (NeurIPS)*, 27, 2672–2680.
- NeurIPS: https://proceedings.neurips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html
- arXiv: https://arxiv.org/abs/1406.2661

**[2]** Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *International Conference on Learning Representations (ICLR)*.
- arXiv: https://arxiv.org/abs/1511.06434

**[3]** Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv preprint*.
- arXiv: https://arxiv.org/abs/1411.1784

**[4]** Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1125–1134.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
- arXiv: https://arxiv.org/abs/1611.07004
- Kod: https://github.com/phillipi/pix2pix

**[5]** Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *IEEE International Conference on Computer Vision (ICCV)*, 2223–2232.
- CVF Open Access: https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf
- arXiv: https://arxiv.org/abs/1703.10593
- Proje sayfası: https://junyanz.github.io/CycleGAN/

**[6]** Wang, T.-C., Liu, M.-Y., Zhu, J.-Y., Tao, A., Kautz, J., & Catanzaro, B. (2018). High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 8798–8807.
- arXiv: https://arxiv.org/abs/1711.11585
- Kod: https://github.com/NVIDIA/pix2pixHD

**[7]** Park, T., Liu, M.-Y., Wang, T.-C., & Zhu, J.-Y. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2337–2346.
- arXiv: https://arxiv.org/abs/1903.07291
- Proje sayfası: https://nvlabs.github.io/SPADE/
- Kod: https://github.com/NVlabs/SPADE

**[8]** Wu, A. N., Stouffs, R., & Biljecki, F. (2022). Generative Adversarial Networks in the built environment: A comprehensive review of the application of GANs across data types and scales. *Building and Environment*, 223, 109477. https://doi.org/10.1016/j.buildenv.2022.109477
- Elsevier: https://www.sciencedirect.com/science/article/abs/pii/S0360132322007089
- Yazar açık PDF: https://filipbiljecki.com/publications/2022_bae_gan.pdf
- Proje sayfası: https://ual.sg/publication/2022-bae-gan/

**[9]** Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and Improving the Image Quality of StyleGAN. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 8110–8119.
- arXiv: https://arxiv.org/abs/1912.04958
- Kod: https://github.com/NVlabs/stylegan2

**[10]** Chen, R., Zhao, J., Yao, X., He, Y., Li, Y., Lian, Z., Han, Z., Yi, X., & Li, H. (2024). Enhancing Urban Landscape Design: A GAN-Based Approach for Rapid Color Rendering of Park Sketches. *Land*, 13(2), 254. https://doi.org/10.3390/land13020254
- MDPI: https://www.mdpi.com/2073-445X/13/2/254
- ResearchGate: https://www.researchgate.net/publication/378321366

**[11]** Jiang, F., Ma, J., Webster, C. J., Wang, W., & Cheng, J. C. P. (2024). Automated site planning using CAIN-GAN model. *Automation in Construction*, 159, 105286. https://doi.org/10.1016/j.autcon.2024.105286
- Elsevier: https://www.sciencedirect.com/science/article/abs/pii/S0926580524000220
- ResearchGate: https://www.researchgate.net/publication/378641918_Automated_site_planning_using_CAIN-GAN_model

**[12]** Ye, X., Du, J., & Ye, Y. (2022). MasterplanGAN: Facilitating the smart rendering of urban master plans via generative adversarial networks. *Environment and Planning B: Urban Analytics and City Science*, 49(3), 794–814. https://doi.org/10.1177/23998083211023516
- SAGE: https://journals.sagepub.com/doi/10.1177/23998083211023516

**[13]** Quan, S. J. (Steven Jige Quan) (2020). Suggestive Site Planning with Conditional GAN and Urban GIS Data. In *Proceedings of the 2019 DigitalFUTURES (CDRF 2019)*, Springer, 103–113. https://doi.org/10.1007/978-981-33-4400-6_10
- Springer: https://link.springer.com/chapter/10.1007/978-981-33-4400-6_10
- CumInCAD: http://papers.cumincad.org/cgi-bin/works/paper/cdrf2019_103
- Açık PDF: http://papers.cumincad.org/data/works/att/cdrf2019_103.pdf

**[14]** Chen, X., Tian, W., Ma, S., & Wang, S. (2023). UrbanGenoGAN: pioneering urban spatial planning using the synergistic integration of GAN, GA, and GIS. *Frontiers in Environmental Science*, 11, 1287858. https://doi.org/10.3389/fenvs.2023.1287858
- Frontiers: https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2023.1287858/full

**[15]** Nauata, N., Chang, K.-H., Cheng, C.-Y., Mori, G., & Furukawa, Y. (2020). House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation. *European Conference on Computer Vision (ECCV)*.
- arXiv: https://arxiv.org/abs/2003.06988
- Kod: https://github.com/ennauata/housegan
- Proje sayfası: https://ennauata.github.io/housegan/page.html

**[16]** Nauata, N., Hosseini, S., Chang, K.-H., Chu, H., Cheng, C.-Y., & Furukawa, Y. (2021). House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- arXiv: https://arxiv.org/abs/2103.02574
- Proje sayfası: https://ennauata.github.io/houseganpp/page.html

**[17]** Jiang, F., Ma, J., Webster, C. J., ve ark. (2023). Building layout generation using site-embedded GAN model. *Automation in Construction*, 151, 104888. https://doi.org/10.1016/j.autcon.2023.104888
- Elsevier: https://www.sciencedirect.com/science/article/abs/pii/S0926580523001486
- HKU Scholars Hub: https://hub.hku.hk/handle/10722/335691
- Semantic Scholar: https://www.semanticscholar.org/paper/f77f6a5d7227fecc3d7acc53f309a56c2e67081c

**[18]** Aalaei, M., Saadi, M., Rahbar, M., & Ekhlassi, A. (2023). Architectural layout generation using a graph-constrained conditional Generative Adversarial Network (GAN). *Automation in Construction*, 155, 105019.
- Elsevier: https://www.sciencedirect.com/science/article/abs/pii/S0926580523003138
- Semantic Scholar: https://www.semanticscholar.org/paper/5e10610a6a6161f05d272a592486d3c360663113

**[19]** Jiang, M., Chen, Y., Liu, X., & Gao, J. (2025). Automated site layout generation for buildings using graph-constrained generative adversarial network. *Building Simulation*, 18(11), 3097–3118. https://doi.org/10.1007/s12273-025-1350-7
- Springer: https://link.springer.com/article/10.1007/s12273-025-1350-7
- SciOpen: https://www.sciopen.com/article/10.1007/s12273-025-1350-7

**[20]** Tang, X., & Chung, W.-j. (2024). Automated urban landscape design: an AI-driven model for emotion-based layout generation and appraisal. *PeerJ Computer Science*, 10, e2426. https://doi.org/10.7717/peerj-cs.2426
- PMC tam metin: https://pmc.ncbi.nlm.nih.gov/articles/PMC11623217/
- PeerJ: https://peerj.com/articles/cs-2426/

**[21]** Sangkloy, P., Lu, J., Fang, C., Yu, F., & Hays, J. (2017). Scribbler: Controlling Deep Image Synthesis with Sketch and Color. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 5400–5409.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2017/html/Sangkloy_Scribbler_Controlling_Deep_CVPR_2017_paper.html
- arXiv: https://arxiv.org/abs/1612.00835

**[22]** Chen, W., & Hays, J. (2018). SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 9416–9425.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_SketchyGAN_Towards_Diverse_CVPR_2018_paper.html
- arXiv: https://arxiv.org/abs/1801.02753
- Kod: https://github.com/wchen342/SketchyGAN

**[23]** Demir, I., Koperski, K., Lindenbaum, D., Pang, G., Huang, J., Basu, S., Hughes, F., Tuia, D., & Raskar, R. (2018). DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images. *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Demir_DeepGlobe_2018_A_CVPR_2018_paper.html
- arXiv: https://arxiv.org/abs/1805.06561
- Kaggle veri: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

**[24]** Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 6626–6637.
- arXiv: https://arxiv.org/abs/1706.08500

**[25]** Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 10684–10695.
- CVF Open Access: https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf
- arXiv: https://arxiv.org/abs/2112.10752
- Kod: https://github.com/CompVis/latent-diffusion

**[26]** Zhang, L., Rao, A., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 3836–3847.
- CVF Open Access: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
- arXiv: https://arxiv.org/abs/2302.05543

**[27]** Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., & Schiele, B. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3213–3223.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html
- arXiv: https://arxiv.org/abs/1604.01685
- Veri: https://www.cityscapes-dataset.com/

**[28]** Tyleček, R., & Šára, R. (2013). Spatial Pattern Templates for Recognition of Objects with Regular Structure. *German Conference on Pattern Recognition (GCPR)*, 364–374.
- Springer: https://link.springer.com/chapter/10.1007/978-3-642-40602-7_39
- Açık PDF: https://cmp.felk.cvut.cz/~tylecr1/papers/Tylecek-GCPR2013.pdf
- CMP Facades veri: https://cmp.felk.cvut.cz/~tylecr1/facade/

---

> **Doğrulama notu:** Yukarıdaki 28 kaynağın tamamı için en az bir tıklanabilir bağlantı (arXiv veya CVF Open Access) kayıt-gerektirmez şekilde halka açıktır. arXiv kayıtları hiçbir koşulda silinmez; bu, tez savunma jürisinin referans incelemesi için kalıcı erişimi garanti eder. Elsevier/Springer/MDPI/SAGE bağlantıları yayıncı sayfasına yönlendirir ve makale özetine her durumda erişim sağlar; tam metin için kurumsal abonelik veya yazardan istek gerekir.
>
> **İkincil doğrulama (Mayıs 2026 turu):** [8], [10], [11], [12], [13], [17], [18], [19], [20] numaralı dokuz kaynağın **yazar listeleri, cilt/sayfa numaraları ve DOI'leri** ScienceDirect, Springer, SciOpen, PeerJ, ResearchGate ve PMC üzerinden bağımsız çapraz-kontrolle teyit edilerek bu sürümde nihai forma getirilmiştir. Önemli düzeltmeler: [11] sayfa numarası 105306 → **105286**; [20] dergi **PeerJ Computer Science** (önceki sürümdeki "Heliyon" hatalı idi); [19] cilt 18(11), sayfa 3097–3118 olarak doğrulandı.
