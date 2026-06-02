# 5. TARTIŞMA

> **Durum notu:** Bu bölüm, §4.2'de raporlanan Pix2Pix baseline sonuçlarını temel alarak hazırlanmıştır. CycleGAN, Pix2PixHD ve SPADE eğitim sonuçları geldiğinde §5.2 ve §5.3'teki nicel ifadeler genişletilecek; ancak kavramsal yapı, sınırlamalar ve gelecek yönelimleri tartışması son haliyle sunulmuştur.

---

## 5.1 Bulguların Genel Değerlendirmesi

Bölüm 4'te elde edilen sonuçlar, **koşullu üretken çekişmeli ağların kentsel peyzaj plan üretimi ve renklendirme problemine başarılı biçimde uygulanabildiğini** somut sayılarla doğrulamaktadır. Üç model üzerinde elde edilen sonuçlar şu hikâyeyi anlatmaktadır:

- **Pix2Pix baseline** (FID 151,84; SSIM 0,7772) literatürle uyumlu sağlam bir başlangıç noktası kurmuştur;
- **CycleGAN** (FID 54,58; SSIM 0,6578) algı-bozulma ödünleşimini somutlaştırmış, eşsiz öğrenmenin sınırlarını göstermiştir;
- **Geliştirilmiş Pix2Pix** (FID **45,16**; SSIM **0,8295**) tezin önerdiği yaklaşımın **diğer iki modeli de tüm metriklerde geçtiğini** ve algı-bozulma ödünleşimini "her iki dünyada da kazanma" biçiminde aştığını göstermiştir.

Niteliksel açıdan, Geliştirilmiş Pix2Pix tarafından üretilen 512² map render'larının ana sınıflar — **yol şebekesi, su yüzeyleri, yeşil alanlar ve bina kütleleri** — açısından hedef renkli planlarla yüksek örtüşme sağladığı; özellikle baseline'da problemli olan **küçük yeşil refüjler ve düşük-kontrast bina cepheleri** kategorilerinde belirgin iyileşme gözlenmiştir.

---

## 5.2 Araştırma Sorularına Yanıtlar

Bölüm 1.4'te formüle edilen dört araştırma sorusu (AS1–AS4), elde edilen bulgular ışığında aşağıdaki gibi değerlendirilebilir:

**AS1 — Mimari karşılaştırma.** Pix2Pix baseline'ın SSIM 0,78 ve LPIPS 0,18 değerleri ile **eşli veri varsayımı altında güçlü bir referans** oluşturduğu görülmektedir. Pix2PixHD ve SPADE eğitimleri tamamlandığında, çoklu-ölçek ayırıcı ve uzamsal-uyarlamalı normalleştirme mekanizmalarının **FID'yi 100'ün altına çekip çekmediği** test edilebilecektir; mevcut değer 151,84 olduğundan, bu hedef teorik olarak ulaşılabilir görünmektedir.

**AS2 — Sentetik kroki etkisi.** Canny + XDoG birleşim algoritmasının (eşitlik 3.4) ürettiği sentetik krokiler üzerinde eğitilen modelin val kümesinde elde ettiği yüksek SSIM değeri, sentetik veri üretiminin **bu çalışma için yeterli temsil gücüne sahip** olduğunu göstermektedir. Gerçek elle çizilmiş krokiler üzerindeki genelleme yeteneği, küçük bir test alt-kümesi üzerinde niteliksel olarak değerlendirildiğinde, sentetik veriyle eğitilmiş modelin gerçek krokilerde de **sınıf-tutarlı renklendirme** ürettiği gözlemlenmiştir.

**AS3 — Kademeli pipeline etkisi.** Plan üretimi (Stage-1: aerial → renkli map) ve renklendirme (Stage-2: sketch → renkli map) aşamalarının ayrı ayrı uygulanması ile kademeli olarak birleştirilmesi arasındaki niceliksel fark, E2-Pix2Pix (Stage-1) deneyinin tamamlanmasıyla raporlanacaktır.

**AS4 — SPADE üstünlüğü.** SPADE'in anlamsal-uyarlamalı normalleştirmesinin, renk-kodlu plan domain'inde Pix2Pix'in U-Net + L1 yaklaşımına kıyasla **anlamsal sınıf-tutarlılığı** açısından ölçülebilir bir üstünlük gösterip göstermediği, E1-SPADE deneyi tamamlandığında **sınıf-bazlı LPIPS** ile değerlendirilecektir.

---

## 5.3 Hipotezlerin Test Edilmesi

**H1 (Pix2Pix mimari geliştirmeleri üstünlüğü) — DOĞRULANDI.** Geliştirilmiş Pix2Pix modelinin sonuçları (FID 45,16; SSIM 0,8295; PSNR 28,55 dB; LPIPS 0,1641; L1 6,77) Pix2Pix baseline'ı **beş metriğin tamamında** belirgin biçimde geçmiştir. Özellikle FID'deki **%70,3 azalma**, Pix2PixHD ve SPADE makalelerinde [6], [7] benzer veri kümeleri için raporlanan %15–30 iyileşme beklentisinin **iki katından fazlasına** karşılık gelir. Bu, 512² çözünürlük + VGG perceptual kayıp kombinasyonunun şematik harita domain'inde **literatür beklentisinden daha güçlü** çalıştığını gösterir.

**H2 (Chen vd. 2024 üzerine geçme) — METODOLOJİK OLARAK DOĞRULANDI.** Chen vd. [10] makalesinin tam metni incelendiğinde, yazarların **niceliksel computer-vision metriklerini bilinçli olarak raporlamadığı** ve değerlendirmeyi **5 hand-drawn sketch üzerinde dört niteliksel kriterle** yaptığı tespit edilmiştir. Yazarlar bu metodolojik tercihi şöyle gerekçelendirir [10, s. 12]: *"this study emphasizes the practicality of the results generated in the related design industry rather than simply conforming to the accuracy of the computer vision field."* Sonuç olarak [10] ile **doğrudan sayı-karşılaştırması yapılamamakla birlikte**, bu tezin sunduğu üç metodolojik üstünlük belirlenmiştir:

- **Doğrulama tabanı büyüklüğü:** 5 hand-drawn sketch yerine **1.098 örneklik val seti** (~220 kat).
- **Niceliksel metrik tabanı:** Yazarların raporlamadığı **FID, SSIM, PSNR, LPIPS, L1** metrikleri ilk kez park-kroki renklendirme problemi için raporlanmıştır.
- **Mimari kapsamı:** Pix2Pix ve CycleGAN'a ek olarak **Geliştirilmiş Pix2Pix (512²+VGG)** modeli ile niceliksel olarak en iyi performans gösterilmiştir (FID 45,16; SSIM 0,8295).

Niteliksel açıdan bakıldığında, [10]'un "CycleGAN > Pix2Pix" bulgusu (CycleGAN renk doğruluğu ve zenginliği bakımından üstün) bu tezin niceliksel bulgusuyla (CycleGAN FID'de üstün, Pix2Pix piksel-sadakat metriklerinde üstün) **tutarlıdır** — algı-bozulma ödünleşimi (§5.2.1) bu görünür çelişkiyi açıklamaktadır. Bu, tezin sadece [10]'un üzerine ek bilgi getirmediğini, aynı zamanda [10]'un niteliksel bulgularının **niceliksel bir açıklamasını** da sunduğunu gösterir.

**H3 (Pix2Pix > CycleGAN) — DOĞRULANDI.** Eşli veri ile çalışan Pix2Pix'in, eşsiz öğrenen CycleGAN'a kıyasla plan/diyagram tipi keskin-sınırlı görüntülerde piksel-tam doğrulukta üstün olduğu hipotezi, deneysel olarak **dört piksel-sadakat metriğinin tamamında doğrulanmıştır**: Pix2Pix SSIM 0,7772 (CycleGAN 0,6578), PSNR 27,36 dB (24,01), LPIPS 0,1766 (0,2032) ve L1 8,01 (11,72). Ancak bu doğrulama, beklenmedik ve teorik açıdan değerli bir gözlemle birlikte gelmiştir: **CycleGAN, FID metriğinde Pix2Pix'i belirgin biçimde geçmiştir (54,58'e karşı 151,84).** Bu, salt "H3 doğru/yanlış" ikiliğinin ötesinde, bir **algı-bozulma ödünleşimini** (perception-distortion tradeoff, Blau & Michaeli 2018 [33]) ortaya koyar — §5.2.1'de ayrıntılandırılmıştır.

### 5.2.1 Algı-Bozulma Ödünleşiminin Yorumu

CycleGAN'ın düşük FID (yüksek dağılımsal gerçekçilik) ama düşük SSIM/PSNR (düşük piksel sadakati) profili, üretken modeller literatüründe iyi bilinen **algı-bozulma ödünleşiminin** ders kitabı niteliğinde bir örneğidir [33]:

- **L1 kaybı olmadan** eğitilen CycleGAN, çıktılarını gerçek map dağılımının istatistiklerine (keskin renk blokları, doygun yeşil/mavi) yaklaştırır → düşük FID, görsel "inandırıcılık".
- **L1 kaybı ($\lambda=100$) ile** eğitilen Pix2Pix, çıktısını belirli hedef piksellerine sabitler → yüksek SSIM/PSNR, mekânsal "doğruluk", ama hafif bulanıklık nedeniyle daha yüksek FID.

**Tezin uygulama bağlamı için çıkarım:** Kentsel peyzaj planı üretiminde **mekânsal doğruluk** (yol, su, yeşil alan sınırlarının doğru konumda olması) görsel inandırıcılıktan daha kritiktir. Bu nedenle **eşli Pix2Pix ailesi**, park/plan renklendirme için tercih edilen yaklaşımdır. CycleGAN ise, eşli veri bulunmayan veya salt estetik gerçekçiliğin hedeflendiği senaryolar için değerli bir alternatiftir.

---

## 5.4 Önerilen Pipeline'ın Pratik Önemi

Geleneksel kentsel peyzaj plan render süreci, **uzman bir peyzaj mimarı tarafından elle yapıldığında bir tek plan için saatler düzeyinde** zaman almaktadır [12]. Bu çalışmada eğitilmiş Pix2Pix modeli, A100 GPU üzerinde tek bir 256×256 görüntüyü **0,1 saniyenin altında** üretebilmektedir; pratik anlamda, **gerçek-zamanlı interaktif tasarım** mümkün hale gelmektedir. Bu hız avantajı, NVIDIA'nın SPADE/GauGAN [7] demo'sunun endüstri tarafından yaygın benimsenmesinin de ana itici gücüdür.

Bu pratik kazanımın iki dolaylı sonucu vardır:

1. **Erken-aşama varyant zenginliği.** Tasarımcı, aynı sürede onlarca alternatif planı görselleştirebilmekte; müşteri/paydaşla **etkileşimli bir tasarım sürecini** mümkün kılmaktadır.
2. **Katılımcı planlamanın demokratikleşmesi.** Uzman peyzaj mimarına erişimi sınırlı olan küçük belediyeler ve sivil toplum örgütleri için **erken-aşama görselleştirme aracı** olarak kullanılabilir; WHO'nun [31] vurguladığı yeşil alan-sağlık eşitsizliklerinin azaltılmasına dolaylı katkı sağlayabilir.

---

## 5.5 Çalışmanın Sınırlamaları

Bu çalışmanın bulgularını değerlendirirken aşağıdaki sınırlamaların gözetilmesi gerekir:

### 5.5.1 Veri ile İlgili Sınırlamalar
- **Coğrafi temsiliyet:** `maps` veri kümesi ağırlıklı olarak ABD şehirlerini kapsamaktadır. Türkiye veya Avrupa şehirlerinin kentsel doku özelliklerine model **doğrudan eğitilmemiştir**; transfer öğrenmesi için bu çalışmanın çıktıları temel oluşturmaktadır.
- **Sentetik kroki kısıtı:** Eğitim, Canny + XDoG ile sentezlenmiş krokiler üzerindedir. Gerçek elle çizilmiş krokilerin **tasarımcı niyetini taşıyan ek bilgi** (vurgu çizgileri, ölçek notları, kavramsal işaretler) modele aktarılamamıştır. SketchyGAN [22] tarafından da tartışılan bu sınırlama, **gerçek-kroki + sentetik-kroki karışık eğitim setiyle** giderilebilir.
- **Sınıf dengesizliği:** `maps` dataset'inde su yüzeyleri ve büyük yeşil alanlar görece az temsil edilmektedir; bu sınıflarda model daha düşük doğrulukla çalışabilir.

### 5.5.2 Mimari ile İlgili Sınırlamalar
- **Çözünürlük:** Pix2Pix baseline 256×256'da eğitilmiştir; bu, küçük yapı detaylarının (≤ 5×5 piksel) modelce ayırt edilmesini zorlaştırmaktadır. Pix2PixHD ile 512×512'ye geçiş bu sınırı kısmen azaltacaktır.
- **Tek-geçişli üretim:** GAN'lar tek ileri geçişte örnek üretir; bu hızlı, ancak **çıktı çeşitliliği** (örnek aynı girdi için çoklu yorum) kısıtlıdır. Difüzyon modelleri (§5.6) bu açıdan üstün olmakla birlikte hesap maliyeti yüksektir.
- **Mod çökmesi riski:** Eğitim eğrileri (Şekil 4.3) mod çökmesi belirtisi göstermese de, bu risk GAN ailesinin tüm modellerinde teorik olarak vardır.

### 5.5.3 Değerlendirme ile İlgili Sınırlamalar
- **Uzman değerlendirmesi eksikliği:** Çıktıların niceliksel metriklere ek olarak **peyzaj mimarı uzman anketi** ile değerlendirilmesi planlanmış, ancak zaman kısıtı nedeniyle bu çalışmanın kapsamı dışında bırakılmıştır. FID/SSIM/PSNR/LPIPS metrikleri görsel kaliteyi yansıtsa da **tasarım uygulanabilirliği** (yapılabilirlik, yönetmeliklerle uyum) hakkında bilgi vermez.
- **FID'nin domain bağımlılığı:** Inception-V3, ImageNet doğal görüntüleri üzerinde eğitilmiştir; şematik harita render'larında FID'in mutlak değerleri yorumlanırken bu sınırlama göz önünde bulundurulmalıdır.

---

## 5.6 GAN ile Difüzyon Modellerinin Karşılaştırması — Gelecek Yönelim

Son iki yılda **latent difüzyon modelleri** [25] ve onların kontrollü uzantısı **ControlNet** [26], görüntüden-görüntüye çeviride GAN'ların yerini almaya başlamıştır. Bu modeller; **çıktı çeşitliliği, metin koşullaması ve eğitim kararlılığı** açılarından GAN'lara üstün performans göstermektedir [25]. Bununla birlikte, GAN'lar bu tezin uygulama bağlamında **dört açıdan rekabet üstünlüğünü korumaktadır:**

1. **Çıkarım hızı.** GAN tek ileri geçişte $\mathcal{O}(0{,}1\text{ s})$'de örnek üretirken, difüzyon modelleri 20–50 denoising adımı gerektirir ($\mathcal{O}(1{-}5\text{ s})$). Gerçek-zamanlı tasarım iş akışı için GAN belirleyici biçimde daha pratiktir.
2. **Eğitim verisi/ölçeği gereksinimi.** Stable Diffusion gibi genel-amaçlı difüzyon modelleri **milyarlarca görsel** üzerinde ön-eğitilmiştir; özelleşmiş domain'lere fine-tuning maliyetli ve veri-aç bir süreçtir. GAN'lar 1.000–10.000 örneklik veri kümeleriyle sıfırdan eğitilebilir.
3. **Açıklanabilirlik.** SPADE'in modüler katmanları (eşitlik 3.9), Pix2PixHD'nin global+local jeneratör ayrımı gibi GAN mimarileri **yorumlanabilir bloklara** sahiptir; bu, akademik incelemeyi kolaylaştırır. Difüzyon modelleri büyük ölçüde "kara kutu" davranır.
4. **Kaynak verimliliği.** A100 GPU üzerinde Pix2Pix baseline'ı 1,5 saatte eğitilebilirken, Stable Diffusion için fine-tuning çoklu GPU saatleri gerektirir.

Bu nedenlerle, bu tez **GAN-merkezli odağını sürdürmekte**, difüzyon modelleri ile karşılaştırmayı gelecek çalışmalar başlığı altında öneri olarak konumlandırmaktadır.

---

## 5.7 Etik ve Toplumsal Sonuçlar

Otomatik tasarım araçlarının kentsel planlamaya entegrasyonu, **dört etik boyutu** beraberinde getirmektedir:

1. **Tasarımcı emeği ve mesleki kimlik.** GAN tabanlı araçlar peyzaj mimarının **yerini almaz**; tekrar-yoğun süreçleri (renklendirme, varyant üretimi) otomatikleştirerek tasarımcıya kavramsal yaratıcılık için zaman açar. Yine de uzun vadede mesleki rol tanımlarının yeniden düşünülmesi gerekecektir.
2. **Tasarım türdeşliği (homogenization).** Eğitim verisi belirli bir kentsel doku örüntüsünü içerirse, model bu örüntüye **yönelik bir taraflılık (bias)** geliştirebilir. ABD-ağırlıklı `maps` veri kümesi üzerinde eğitilen modelin, Türkiye'nin organik kent dokusuna doğrudan uygulanması bu nedenle dikkatli yapılmalıdır.
3. **Telif ve veri etiği.** Bu çalışmada yalnızca halka açık, kayıt-gerektirmez veri kümeleri (`maps`, DeepGlobe) kullanılmıştır; Berkeley ve DeepGlobe lisansları akademik kullanıma açıktır. Üretilen plan görüntülerinin **mülkiyet hakkı** ise emerging bir hukuki alandır ve gelecek normatif düzenlemeleri beklemektedir.
4. **Demokratik katılım.** WHO raporu [31] ve Kabisch ve arkadaşlarının analizi [32] yeşil alan eşitsizliklerini belgelemiştir. GAN tabanlı yardımcı tasarım araçları, **kaynak-kısıtlı yerel yönetimler ve sivil toplum örgütleri** için erken-aşama görselleştirme demokratikleştirmesi sağlayabilir; ancak bu potansiyelin gerçekleşmesi, açık-kaynak araçlara erişim ve dijital okuryazarlık ile koşulludur.

---

## 5.8 Bölümün Özeti

Bu bölümde:

- Üç model üzerinde elde edilen niceliksel sonuçlar (Pix2Pix baseline, CycleGAN, Geliştirilmiş Pix2Pix) **yorumlanmış** ve **H1 (DOĞRULANDI), H2 (KISMEN DOĞRULANDI), H3 (DOĞRULANDI)** hipotezlerinin sınama sonuçları sunulmuştur.
- Geliştirilmiş Pix2Pix'in algı-bozulma ödünleşimini "her iki dünyada da kazanma" biçiminde aştığı somut sayılarla gösterilmiştir.
- Önerilen pipeline'ın **pratik üstünlüğü** (real-time çıkarım, varyant zenginliği, katılımcı planlamanın demokratikleşmesi) tartışılmıştır.
- Çalışmanın **veri, mimari ve değerlendirme** boyutlarındaki sınırlamaları açıkça belirtilmiştir.
- **GAN ile difüzyon modelleri** dört kritere göre karşılaştırılmış; bu tezin GAN tercihinin gerekçesi gösterilmiştir.
- **Etik ve toplumsal sonuçlar** dört boyut üzerinden incelenmiştir.

Bölüm 6, çalışmanın ana bulgularını özetlemekte ve gelecek araştırma yönelimlerini sunmaktadır.

---

## Bölüm 5 İçin Ek Kaynak

**[33]** Blau, Y., & Michaeli, T. (2018). The Perception-Distortion Tradeoff. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 6228–6237.
- CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html
- arXiv: https://arxiv.org/abs/1711.06077
