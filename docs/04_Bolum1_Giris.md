# 1. GİRİŞ

> **Bilgi notu:** Bu bölümde verilen tüm referanslar Bölüm 2'deki numaralı kaynak listesini takip eder. [29]–[32] arası yeni eklenen sosyal-bilimsel referanslar bölüm sonuna eklenmiştir; her biri kayıt-gerektirmez, kalıcı bağlantılarla doğrulanmıştır.

---

## 1.1 Çalışmanın Arka Planı

Yirmi birinci yüzyıl, insanlık tarihinin **en yoğun kentleşme dönemi** olarak nitelendirilmektedir. Birleşmiş Milletler Ekonomik ve Sosyal İşler Departmanı'nın (UN DESA) *World Urbanization Prospects 2018* raporuna göre dünya kentsel nüfusu 1950'de 751 milyon iken 2018'de 4,2 milyara ulaşmış; 2050'ye kadar bu rakamın **6,7 milyara çıkması** ve dünya toplam nüfusunun %68'inin kentlerde yaşaması beklenmektedir [29]. Seto, Güneralp ve Hutyra'nın *Proceedings of the National Academy of Sciences (PNAS)* dergisinde yayımladığı uzun-vadeli mekânsal projeksiyonlar ise küresel kentsel arazi örtüsünün **2030 yılına kadar 2000 yılına kıyasla yaklaşık üç katına** çıkacağını ve bu genişlemenin biyoçeşitlilik sıcak noktaları üzerinde doğrudan ekolojik baskı oluşturacağını ortaya koymaktadır [30].

Bu hızlı kentleşme süreci üç temel kentsel zorluğu eş zamanlı olarak gündeme getirmektedir:

1. **Aşırı kentleşme ve fiziksel-mekânsal kalite kaybı:** Düzensiz, plansız veya hızlı planlanan kentsel büyüme; insan ölçeği, hava kalitesi ve sosyal etkileşim olanakları açısından yaşam kalitesini düşürmektedir.
2. **İklim değişikliği ve çevresel bozulma:** Kentsel ısı adası etkisi, yağış rejimi değişiklikleri ve karbon emisyonları, sürdürülebilir kentsel planlamayı zorunlu kılmaktadır.
3. **Kentsel yeşil alanların erişilebilirliği:** Dünya Sağlık Örgütü'nün (WHO) Avrupa Bürosu tarafından 2017'de yayımlanan kapsamlı derleme raporu, kentsel yeşil alanların **fiziksel sağlık (obezite, kardiyovasküler hastalıklar), zihinsel sağlık (stres, depresyon) ve sosyal eşitlik** üzerinde nedensel olumlu etkilerini sistematik biçimde belgelemiştir [31]. Buna karşın Kabisch ve arkadaşları'nın 299 Avrupa şehri üzerinde yaptığı analiz, özellikle **Güney Avrupa şehirlerinde yeşil alan erişilebilirliğinin ortalama altında** kaldığını göstermektedir [32].

Bu üç zorluğun ortak paydası, **kentsel tasarımın yalnızca estetik bir uğraşı değil, halk sağlığı, ekolojik denge ve sosyal adalet için temel bir araç** olmasıdır. Geleneksel kentsel tasarım sürecinde plan üretimi; saha analizi, eskiz, varyant geliştirme ve renkli sunum aşamalarından oluşan; **emek-yoğun, zaman alıcı ve uzman bağımlı** bir döngü olarak yürütülmektedir. Otomatik veya yarı-otomatik plan üretim yöntemleri, bu döngüyü hızlandırarak tasarımcının daha fazla varyant denemesini, daha fazla paydaş katılımını mümkün kılmaktadır [11], [12].

---

## 1.2 Problem Tanımı

Bu tezin ele aldığı problem iki birbirine bağlı alt görevden oluşmaktadır:

**(P1) Kentsel Peyzaj Plan Üretimi.** Belirli bir alan sınırı, çevresel bağlam (yol, mevcut yapı, su yüzeyleri) ve fonksiyonel gereksinim verildiğinde; bu girdilerden **anlamsal olarak makul, görsel olarak tutarlı bir kentsel peyzaj planı** otomatik olarak üretilebilir mi?

**(P2) Kroki Renklendirme.** Tasarımcının erken aşamada ürettiği siyah-beyaz kroki/eskiz çizimleri, peyzaj mimarlığı disiplininin estetik konvansiyonlarına uygun **renkli, sunum-hazır plana** otomatik olarak nasıl dönüştürülebilir?

Bu iki alt görev, **görüntüden-görüntüye çeviri (image-to-image translation)** alt alanına aittir ve son on yılda **üretken çekişmeli ağlar (Generative Adversarial Networks — GAN)** [1], [3], [4] tarafından önemli ölçüde çözülmüştür. Bununla birlikte, kentsel peyzaj alanına özgü uygulamalar hâlâ **azlık ve dağınıklık** göstermektedir.

Literatürde mevcut çalışmaların kritik sınırlamaları şunlardır:

- **Veri ölçeği problemi:** Park-spesifik renklendirme çalışmaları çok küçük dataset'ler kullanmıştır; Chen ve arkadaşlarının doğrudan bu tezin alt başlığı ile örtüşen çalışması [10] yalnızca 152 eşli görüntü ile eğitilmiştir.
- **Mimari sınırlılık:** Aynı çalışma yalnızca temel Pix2Pix [4] ve CycleGAN [5] mimarilerini denemiş; yüksek-çözünürlüklü Pix2PixHD [6] veya anlamsal-uyarlamalı SPADE [7] gibi son nesil modeller park-sketch renklendirme problemine **henüz uygulanmamıştır**.
- **Pipeline bütünlüğü eksikliği:** CAIN-GAN [11] ve MasterplanGAN [12] gibi çalışmalar yalnızca plan üretimine; Chen ve arkadaşları [10] yalnızca renklendirmeye odaklanmıştır. **İki aşamayı entegre eden uçtan-uca (end-to-end) bir pipeline yoktur.**
- **Açık veri kullanımı sınırı:** Mevcut peyzaj çalışmaları çoğunlukla kapalı/özel datasetler kullanmaktadır; bu, sonuçların yeniden üretilebilirliğini (reproducibility) güçleştirmektedir.

---

## 1.3 Motivasyon ve Önem

Bu tezin motivasyonu üç farklı düzlemde ifade edilebilir:

### 1.3.1 Akademik Düzlem
GAN tabanlı görüntü çevirisi, 2014'teki Goodfellow ve arkadaşlarının çığır açan çalışmasından [1] bu yana hızla olgunlaşmış; mimari ve şehircilik disiplinlerinde son dört yılda Elsevier *Automation in Construction*, Springer *Building Simulation*, SAGE *Environment and Planning B* ve MDPI *Land* gibi etki faktörü yüksek dergilerde özelleşmiş çalışmalar artarak yer almaktadır [10]–[19]. Bu olgunlaşmaya rağmen **park ve bahçe ölçeğinde** sistematik bir karşılaştırmalı analiz literatürde mevcut değildir; bu boşluk Bölüm 2.8'de detaylı tartışılmıştır.

### 1.3.2 Pratik Düzlem
Geleneksel kentsel peyzaj plan render süreci, peyzaj mimarı tarafından elle yapıldığında **bir tek plan için birkaç saat ile birkaç gün** arasında zaman almaktadır. GAN tabanlı renderlama, eğitilmiş model üzerinde bir tek ileri geçiş (forward pass) ile **saniyeler içinde** çıktı üretebilmektedir. Bu, tasarımcının erken-aşamada çok daha fazla varyant denemesine ve müşteri/paydaş ile etkileşimli olarak çalışmasına imkân tanır [10], [12]. SPADE'in NVIDIA tarafından geliştirilen interaktif kullanıcı arayüzü olan GauGAN, bu pratiğin endüstride benimsenebilirliğini somut olarak göstermiştir [7].

### 1.3.3 Toplumsal-Çevresel Düzlem
WHO'nun yeşil alan-sağlık ilişkisini sistematik olarak belgelediği rapor [31] ve Kabisch ve arkadaşlarının yeşil alan eşitsizliğini ortaya koyan analizi [32] göz önünde bulundurulduğunda; **yeşil alan tasarım süreçlerini hızlandıran ve demokratikleştiren** araçların kentsel sağlık eşitsizliklerinin azaltılmasına dolaylı katkı sunabileceği değerlendirilebilir. GAN tabanlı yardımcı tasarım araçları, küçük belediyeler ve sivil toplum örgütleri için uzman peyzaj mimarına erişimi olmayan durumlarda **erken-aşama görselleştirme** sağlayarak katılımcı planlamayı kolaylaştırabilir.

---

## 1.4 Çalışmanın Amacı ve Araştırma Soruları

### 1.4.1 Amaç
Bu tezin temel amacı; **koşullu üretken çekişmeli ağ (cGAN) mimarileri kullanarak kentsel peyzaj — özellikle park ve bahçe — planlarının otomatik üretimini ve siyah-beyaz krokilerin renklendirilmesini gerçekleştiren, halka açık dataset'lere dayanan, yeniden üretilebilir bir uçtan-uca yazılım çerçevesi geliştirmek** ve dört farklı GAN mimarisinin (Pix2Pix, CycleGAN, Pix2PixHD, SPADE) bu problem üzerindeki performansını standart metriklerle karşılaştırmaktır.

### 1.4.2 Araştırma Soruları
Yukarıdaki amaç çerçevesinde aşağıdaki dört araştırma sorusu (AS) ele alınmaktadır:

- **AS1.** Halka açık `maps` (Berkeley) [4] dataset'i üzerinde Pix2Pix, CycleGAN, Pix2PixHD ve SPADE mimarilerinin **sketch → renkli plan** çevirisindeki performansları FID, SSIM, PSNR ve LPIPS metrikleri açısından nasıl farklılaşmaktadır?
- **AS2.** Canny + XDoG tabanlı **sentetik kroki üretiminin**, gerçek elle çizilmiş kroki test alt-kümesinde, modelin genelleme (generalization) yeteneği üzerinde etkisi nedir?
- **AS3.** Plan üretimi (Stage-1: aerial → semantic map) ve renklendirme (Stage-2: sketch → semantic map) aşamalarının **kademeli (cascaded) eğitimi**, iki aşamayı ayrı ayrı uygulamaya kıyasla nicel ve nitel olarak ne tür kazanımlar sağlar?
- **AS4.** SPADE'in anlamsal-uyarlamalı normalleştirmesi [7], park-sketch domain'inde Pix2Pix'in U-Net + L1 yaklaşımına kıyasla **anlamsal sınıf-tutarlılığı (semantic consistency)** açısından ölçülebilir bir üstünlük gösterir mi?

### 1.4.3 Hipotezler
Yukarıdaki sorulara karşılık aşağıdaki hipotezler test edilmektedir:

- **H1.** Pix2PixHD ve SPADE, çoklu-ölçek ayırıcı ve uzamsal-uyarlamalı normalleştirme avantajları sayesinde Pix2Pix ve CycleGAN'a kıyasla **istatistiksel olarak anlamlı şekilde daha düşük FID ve daha yüksek SSIM** üretecektir.
- **H2.** GAN-temelli veri çoğaltma + sentetik sketch üretimi birlikte uygulandığında, model değerlendirme metrikleri Chen ve arkadaşlarının [10] raporladığı baseline değerlerini aşacaktır.
- **H3.** Eşli (paired) veri ile çalışan Pix2Pix ailesi, eşsiz (unpaired) öğrenen CycleGAN'a kıyasla plan/diyagram tipi keskin-sınırlı görüntülerde piksel-tam doğrulukta daha üstün olacaktır.

---

## 1.5 Tezin Özgün Katkıları

Bu tez literatüre aşağıdaki **beş özgün katkıyı** sunmaktadır:

1. **Halka açık `maps` dataset'inin park-renklendirme problemine uyarlanması:** Berkeley'in 2194 eşli uydu↔harita çifti, Canny + XDoG filtreleriyle sentetik kroki üretilerek **sketch → renkli plan** problemine adapte edilmiştir. Bu yaklaşım, Chen ve arkadaşlarının [10] 152 örnekli özel datasetine karşılık **yaklaşık 14 kat büyüklükte standart bir benchmark** oluşturmaktadır.

2. **Dört GAN mimarisinin aynı protokol altında sistematik karşılaştırılması:** Pix2Pix [4], CycleGAN [5], Pix2PixHD [6] ve SPADE [7] aynı dataset, aynı eğitim/test bölmesi, aynı augmentation ve aynı metrikler (FID [24], SSIM, PSNR, LPIPS) altında değerlendirilmiştir. Park-sketch renklendirme problemine **Pix2PixHD ve SPADE'in uygulanması literatürde ilktir.**

3. **İki aşamalı (plan üretimi + renklendirme) entegre pipeline:** Stage-1'de aerial görüntüden renk-kodlu plan üretimi, Stage-2'de sketch'ten renkli plan render'ı yapılarak **end-to-end iş akışı** ortaya konmuştur. Bu, CAIN-GAN [11] tarafından önerilen Faz 1/Faz 2 yaklaşımının peyzaj alanına uyarlanmış bir varyantıdır.

4. **DeepGlobe Land Cover [23] üzerinde SPADE değerlendirmesi:** Uzaktan algılama uydu görüntüleri için açık dataset olan DeepGlobe'un 7-sınıflı semantic mask'larından foto-gerçekçi peyzaj görüntüsü üretimi SPADE/GauGAN [7] mimarisi ile yapılmakta; bu, uydu domain'i ile GAN tabanlı peyzaj sentezini köprüleyen az sayıdaki çalışmadan biridir.

5. **Tam yeniden üretilebilir kod tabanı + Colab notebook'u:** Tüm kod, konfigürasyon, dataset hazırlama scriptleri, eğitim, değerlendirme ve görselleştirme adımları MIT lisansı altında halka açık bir GitHub deposunda (`https://github.com/ilker-23/gan-urban-design-v2`) yayımlanmıştır. Google Colab Pro A100 GPU üzerinde tek tıkla çalıştırılabilen referans uygulama sağlanmıştır.

---

## 1.6 Çalışmanın Kapsamı ve Sınırları

### 1.6.1 Kapsam
- **Görüntü modaliteleri:** Yalnızca 2-boyutlu kuş bakışı (top-down) plan ve uydu görüntüleri ele alınmaktadır; perspektif/sokak görünümü kapsam dışıdır.
- **Çözünürlük:** Eğitim 256×256 ve 512×512 piksel görüntü çiftleri üzerinde yapılmıştır; Pix2PixHD'nin önerdiği 2048×1024 çözünürlük A100 GPU bütçesi nedeniyle 512² ile sınırlı tutulmuştur.
- **Mimari ailesi:** Çalışma **GAN-tabanlı** modellerle sınırlıdır. Difüzyon modelleri (Stable Diffusion [25], ControlNet [26]) gelecek çalışmalar olarak Bölüm 5'te tartışılmaktadır.
- **Veri:** Dataset olarak yalnızca halka açık iki kaynak (Berkeley `maps` [4]; DeepGlobe Land Cover [23]) kullanılmıştır.

### 1.6.2 Sınırlamalar
- **Sentetik kroki kısıtı:** Modellerin eğitimi otomatik üretilmiş (Canny + XDoG) sentetik krokiler üzerindedir. Gerçek elle çizilmiş krokiler ile genelleme yeteneği, küçük bir test alt-kümesi üzerinden niteliksel olarak değerlendirilmektedir.
- **Coğrafi bağlam:** Berkeley `maps` dataset'i ağırlıklı olarak Amerika Birleşik Devletleri şehirlerini kapsar. Türkiye veya benzeri ülkelerin kentsel doku özelliklerine model **doğrudan eğitilmemiştir**; transfer öğrenmesi için temel oluşturmaktadır.
- **Uzman değerlendirmesi:** Çıktıların niteliksel değerlendirmesi öncelikle metriklere dayanmaktadır; geniş ölçekli peyzaj mimarı anket çalışması zaman kısıtı nedeniyle bu çalışmanın kapsamı dışında bırakılmıştır.

---

## 1.7 Tezin Organizasyonu

Tezin geri kalan bölümleri aşağıdaki şekilde organize edilmiştir:

- **Bölüm 2 (Literatür Araştırması):** GAN'ların kuramsal temelinden başlayarak; koşullu GAN, görüntüden-görüntüye çeviri, anlamsal görüntü sentezi, kentsel plan üretimi ve kroki renklendirme alt-alanlarındaki son nesil çalışmalar sistematik olarak incelenmekte; literatürdeki boşluk ve tezin konumu açıklanmaktadır.
- **Bölüm 3 (Materyal ve Yöntem):** Kullanılan dataset'ler, sentetik kroki üretim algoritması, dört GAN mimarisinin matematiksel formülasyonu, kayıp fonksiyonları, optimizasyon detayları ve değerlendirme protokolü ayrıntılı olarak sunulmaktadır.
- **Bölüm 4 (Deneysel Sonuçlar):** Dört model üzerinde yapılan deneyler; FID, SSIM, PSNR, LPIPS ve L1 metrikleri ile karşılaştırılmakta; ablation çalışmaları (augmentation etkisi, kayıp ağırlığı $\lambda_{L1}$ etkisi) yapılmakta; niteliksel görsel kıyaslamalar sunulmaktadır.
- **Bölüm 5 (Tartışma):** Bulguların yorumlanması; sınırlamalar; difüzyon modelleri ile karşılaştırma; etik ve toplumsal sonuçlar.
- **Bölüm 6 (Sonuçlar ve Öneriler):** Tezin ana bulgularının özeti ve gelecek araştırma yönelimleri.

---

## Bölüm 1 İçin Ek Kaynaklar

> Bölüm 2'deki [1]–[28] arası numaralandırma korunmuştur; aşağıdaki [29]–[32] yeni sosyal-bilimsel referanslardır.

**[29]** United Nations, Department of Economic and Social Affairs, Population Division (2018). *World Urbanization Prospects: The 2018 Revision*. United Nations, New York.
- Resmi rapor (özet PDF): https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/files/documents/2020/Feb/un_2018_wup_highlights.pdf
- Tam rapor (PDF): https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/files/documents/2020/Jan/un_2018_wup_report.pdf
- Veritabanı portalı: https://population.un.org/wup/

**[30]** Seto, K. C., Güneralp, B., & Hutyra, L. R. (2012). Global forecasts of urban expansion to 2030 and direct impacts on biodiversity and carbon pools. *Proceedings of the National Academy of Sciences (PNAS)*, 109(40), 16083–16088. https://doi.org/10.1073/pnas.1211658109
- PNAS: https://www.pnas.org/doi/10.1073/pnas.1211658109
- PMC tam metin: https://pmc.ncbi.nlm.nih.gov/articles/PMC3479537/

**[31]** World Health Organization Regional Office for Europe (2017). *Urban green space interventions and health: A review of impacts and effectiveness*. WHO Regional Office for Europe, Copenhagen.
- WHO Europe (özet sayfa): https://www.who.int/europe/publications/m/item/urban-green-space-interventions-and-health--a-review-of-impacts-and-effectiveness.-full-report
- Tam rapor (PDF): https://cdn.who.int/media/docs/librariesprovider2/euro-health-topics/environment/2017-urban-green-space-and-health.pdf

**[32]** Kabisch, N., Strohbach, M., Haase, D., & Kronenberg, J. (2016). Urban green space availability in European cities. *Ecological Indicators*, 70, 586–596. https://doi.org/10.1016/j.ecolind.2016.02.029
- Elsevier: https://www.sciencedirect.com/science/article/abs/pii/S1470160X16300504
- HU Berlin Repo: https://fis.hu-berlin.de/converis/portal/detail/Publication/903422418

---

> **Doğrulama notu:** Bu bölümdeki dört yeni referans (UN [29], PNAS/Seto vd. [30], WHO [31], Ecological Indicators/Kabisch vd. [32]) için tüm bağlantılar 31 Mayıs 2026 itibarıyla halka açık ve doğrulanabilir durumdadır. UN raporu doğrudan un.org alt-domain'inden, PNAS makalesi resmi DOI ile, WHO raporu cdn.who.int üzerinden, Kabisch makalesi Elsevier ScienceDirect üzerinden erişilmektedir. PNAS ve PMC kayıtları çift-arşivlenmiştir; Elsevier kayıtları ise DOI üzerinden kalıcıdır.
