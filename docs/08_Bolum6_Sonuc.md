# 6. SONUÇLAR VE ÖNERİLER

## 6.1 Ana Bulguların Özeti

Bu tezin sunduğu çalışma; koşullu üretken çekişmeli ağlar (cGAN) [1], [3], [4] kullanarak kentsel peyzaj — özellikle park ve bahçe — planlarının otomatik üretimi ve siyah-beyaz krokilerin renklendirilmesi problemine **uçtan-uca**, yeniden üretilebilir bir yazılım çerçevesi ortaya koymuştur. Çalışmanın başlıca somut bulguları aşağıda özetlenmiştir:

### 6.1.1 Niceliksel Sonuçlar

- **Pix2Pix baseline (sketch → renkli map):** 1.098 doğrulama çifti üzerinde **FID = 151,84; SSIM = 0,7772; PSNR = 27,36 dB; LPIPS = 0,1766; L1 = 8,01** — Isola vd. [4] şematik domain'leri ile uyumlu literatür baseline'ı.
- **CycleGAN (eşsiz):** **FID = 54,58; SSIM = 0,6578; PSNR = 24,01 dB; LPIPS = 0,2032; L1 = 11,72** — klasik **algı-bozulma ödünleşimini** [33] somut olarak gösteren, FID'de üstün ama piksel sadakatinde geride bir sonuç.
- **Geliştirilmiş Pix2Pix (512² + VGG perceptual) — tezin önerdiği yöntem:** **FID = 45,16; SSIM = 0,8295; PSNR = 28,55 dB; LPIPS = 0,1641; L1 = 6,77.** Beş metriğin tamamında diğer iki modeli de geçerek baseline'a göre **FID'de %70,3 iyileşme, SSIM'de +6,7 puan, LPIPS'te %7,1 azalma** elde etmiştir.
- **Veri ölçeği avantajı:** Bu çalışmanın eğitim dataset'i (2.194 eşli görüntü çifti), park-renklendirme problemine doğrudan uygulanan önceki literatür çalışmasının [10] manuel topladığı verinin **~14 katıdır** (1.699 augment edilmiş örneğinin de ~1,3 katı); doğrulama tabanı (1.098 örnek) ise [10]'un 5 örneklik test setinin **~220 katıdır**.
- **Niceliksel metrik tabanı:** Chen vd. [10] niceliksel computer-vision metrikleri raporlamamış, niteliksel değerlendirmeyi tercih etmiştir; bu tez park-kroki renklendirme problemine ilk niceliksel benchmark zeminini (FID/SSIM/PSNR/LPIPS/L1) kurmuştur.
- **Algı-bozulma ödünleşiminin aşılması:** Geliştirilmiş Pix2Pix, hem FID'de (CycleGAN'ın 54,58 değerini geçerek 45,16) hem de piksel-sadakat metriklerinde Pix2Pix baseline'ı geçerek, üretken model literatüründe nadir görülen **"her iki dünyada da kazanma"** durumunu sergilemiştir.

### 6.1.2 Niteliksel Sonuçlar

- Eğitilmiş Pix2Pix modeli, **yol şebekesi, yeşil alanlar, su yüzeyleri ve bina kütleleri** gibi ana plan sınıflarını **renk-kodu doğru** biçimde üretebilmektedir.
- Modelin sistematik zorlandığı üç vaka (düşük-kontrastlı bina cepheleri, küçük yeşil alan refüjleri, diyagonal yollar) tespit edilmiş; bu vakaların çözünürlük artırımı ve anlamsal mask koşullaması ile **giderilebilir** olduğu öngörülmüştür.

### 6.1.3 Yazılım Çıktısı

Tezin tüm kaynak kodu, konfigürasyonu, dataset hazırlama scriptleri ve Colab notebook'u **MIT lisansı altında** https://github.com/ilker-23/gan-urban-design-v2 deposunda halka açık olarak yayımlanmıştır. Tek bir komut zinciri ile (`scripts/download_data.sh` → `scripts/prepare_sketches.py` → `python -m src.train` → `python -m src.evaluate`) sıfırdan tüm pipeline yeniden üretilebilmektedir.

---

## 6.2 Tezin Özgün Katkıları

Bölüm 1.5'te özetlenen **beş özgün katkı**, deneysel doğrulama ile birlikte aşağıdaki biçimde sonuçlanmıştır:

| # | Katkı | Durum |
|---|-------|-------|
| 1 | Berkeley `maps` veri kümesinin park-renklendirme problemine uyarlanması | ✅ Uygulandı; 2.194 eşli kroki–renkli plan çifti elde edildi. |
| 2 | Pix2Pix, CycleGAN ve Geliştirilmiş Pix2Pix'in aynı protokol altında karşılaştırılması | ✅ Üç model tamamlandı; H1 ve H3 niceliksel olarak doğrulandı. |
| 3 | İki aşamalı (plan üretimi + renklendirme) entegre pipeline | 🟡 Stage-2 (renklendirme) doğrulandı; Stage-1 (sat→map) opsiyonel deney olarak konumlandırıldı. |
| 4 | DeepGlobe Land Cover [23] üzerinde SPADE değerlendirmesi | ⏳ Tez sonrası ek deney olarak gelecek çalışmalara aktarıldı (Y1). |
| 5 | Tam yeniden üretilebilir, MIT lisanslı kod tabanı + Colab notebook'u | ✅ Yayımlandı (https://github.com/ilker-23/gan-urban-design-v2). |

Katkı 1 ve Katkı 5 tam olarak gerçekleşmiş; Katkı 2, 3 ve 4 eğitim aşamasında olan modellerin sonuçlarıyla birlikte tezin teslim haline son şeklini alacaktır.

---

## 6.3 Tezin Sınırları ve Karşılanma Düzeyleri

Bölüm 5.5'te ayrıntılı tartışılan sınırlamalar üç başlık altında özetlenebilir:

- **Veri tarafı:** ABD-ağırlıklı veri kümesi; sentetik kroki kısıtı; sınıf dengesizliği.
- **Mimari tarafı:** 256×256 (Pix2Pix) ve 512×512 (Pix2PixHD/SPADE) çözünürlük üst sınırı; tek-geçişli üretimin çıktı çeşitliliği sınırı.
- **Değerlendirme tarafı:** Uzman anket çalışmasının kapsam dışında bırakılması; FID metriğinin doğal görüntü dağılımına bağımlılığı.

Bu sınırlamalar her biri **gelecek araştırma için belirgin uzantılar** sunmaktadır; §6.4'te ele alınmaktadır.

---

## 6.4 Gelecek Çalışma Yönelimleri

Bu tezin sunduğu bulgular ve sınırlamalar göz önünde bulundurularak aşağıdaki **yedi gelecek araştırma yönelimi** önerilmektedir:

### Y1. Difüzyon Modelleri ile Karşılaştırma
Latent Diffusion Models (LDM) [25] ve ControlNet [26] mimarilerinin aynı `maps` / DeepGlobe veri kümeleri üzerinde fine-tune edilerek FID, SSIM, LPIPS metriklerinde GAN ailesi ile karşılaştırılması. Beklenen: difüzyon modelleri çıktı çeşitliliğinde üstün, hız ve verimlilikte düşük performans gösterecektir.

### Y2. Türkiye Şehirleri için Transfer Öğrenmesi
`maps` üzerinde önceden eğitilmiş Pix2Pix/SPADE modellerinin **Türkiye şehirlerinin** açık uydu görüntüleri (örneğin Copernicus Sentinel-2 veya açık-erişim ortofoto verileri) üzerinde fine-tune edilmesi. Bu, modelin organik/tarihi kent dokusuna uyarlanmasını sağlayabilir.

### Y3. Gerçek Elle Çizilmiş Kroki Veri Toplama
Tasarım okullarıyla işbirliği yapılarak **gerçek elle çizilmiş peyzaj krokisi** veri kümesinin (200–500 örnek) toplanması ve modelin sentetik + gerçek karışık veriyle yeniden eğitilmesi. SketchyGAN [22]'nin önerdiği data augmentation stratejilerinin peyzaj domain'ine adaptasyonu.

### Y4. Çoklu-Modal Koşullama
Sketch'e ek olarak **metinsel etiket** (örneğin "yoğun ağaçlık alan, çocuk oyun alanı, su öğesi") koşullaması ile Scribbler [21] benzeri kullanıcı-yönlendirmeli üretim. Bu, GauGAN [7] arayüzünün doğal bir uzantısıdır.

### Y5. Performans-Bilinçli Üretim (CAIN-GAN Uzantısı)
Jiang ve arkadaşlarının CAIN-GAN [11] yaklaşımının peyzaj alanına genişletilmesi: sürdürülebilirlik göstergeleri (yeşil alan oranı, yaya erişilebilirliği, gölge faktörü) **kayıp fonksiyonuna entegre edilerek** sadece görsel olarak değil **performans olarak da optimize edilmiş** plan üretimi.

### Y6. Çoklu-Amaçlı Optimizasyon
UrbanGenoGAN [14] tarafından önerilen GAN + Genetik Algoritma kombinasyonunun peyzaj problemine uygulanması; üretilen plan adaylarının çoklu-amaçlı (estetik + ekoloji + maliyet) optimizasyona tabi tutulması.

### Y7. Uzman Anketi ve Kullanıcı Çalışması
Peyzaj mimarı uzmanları + son kullanıcılarla (mahalle sakinleri, yerel yönetim temsilcileri) yapılacak **likert ölçekli niceliksel ve niteliksel değerlendirme**; sayısal metrikler ile insan değerlendirmesi arasındaki **uyum/uyumsuzluk** analizi. Bu, otomatik metriklerin (FID, SSIM, vb.) **toplum-merkezli tasarım** kalitesini ne ölçüde yansıttığının ampirik testi olacaktır.

---

## 6.5 Kapanış

Yirmi birinci yüzyılın hızlanan kentleşmesi [29], [30] ve bunun sürdürülebilirlik, halk sağlığı [31] ve sosyal adalet [32] üzerindeki çok-boyutlu etkileri göz önünde bulundurulduğunda; **otomatik, hızlı ve demokratik kentsel peyzaj tasarım araçlarının** geliştirilmesi yalnızca akademik bir uğraş değil, bir **kamu yararı sorumluluğu** olarak değerlendirilmelidir.

Bu tez; üretken çekişmeli ağ literatürünün son on yıllık birikimini [1]–[7] kentsel peyzaj-park ölçeğine taşıyarak, halka açık veri kümeleri üzerinde **yeniden üretilebilir, ücretsiz ve şeffaf** bir yazılım çerçevesi sunmuştur. Pix2Pix baseline'ın elde ettiği niceliksel sonuçlar (SSIM 0,78; FID 152), önerilen pipeline'ın akademik açıdan sağlam, pratik açıdan uygulanabilir olduğunu göstermektedir. Çalışmanın halka açık deposu (https://github.com/ilker-23/gan-urban-design-v2), gelecek araştırmacıların — özellikle yüksek lisans ve doktora öğrencilerinin — kendi veri kümeleri ve coğrafi bağlamlarına uyarlayabilecekleri bir **referans implementasyon** olarak konumlanmaktadır.

Önerilen yedi gelecek araştırma yönelimi (Y1–Y7) ile birlikte değerlendirildiğinde, kentsel peyzaj tasarımının yapay zekâ destekli dönüşümünde bu tezin **bir başlangıç noktası** sağladığı, ancak nihai hedef olan **performans-bilinçli, katılımcı ve toplum-merkezli otomatik tasarım** vizyonunun daha geniş bir araştırma topluluğunun katkısını gerektirdiği vurgulanmalıdır.

---

> **Tez kapanış notu:** Bu çalışmanın tüm araştırma raporları, dataset karar günlüğü, yazılım kodu ve tez bölümleri (`docs/` altında numaralandırılmış halde) https://github.com/ilker-23/gan-urban-design-v2 deposunda kalıcı olarak yayımlanmıştır. Tez teslim tarihindeki son commit hash'i, tez ekinde **EK-A: Yazılım Sürümü** olarak kayıt altına alınacaktır.
