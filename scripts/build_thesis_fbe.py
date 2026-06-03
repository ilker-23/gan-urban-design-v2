"""Tezi Fırat Üniv. FBE şablonu stiliyle Word'e dönüştürür.

Tek master Markdown -> Pandoc (--reference-doc=template) -> .docx
- Şablonun tüm stilleri (font, marj, başlık, paragraf) korunur
- Denklemler native OMML matematik nesnesi olarak gömülür
- Görseller markdown image syntax'ıyla bölüm içine yerleştirilir
- Kaynaklar TEK liste halinde tezin SONUNDA verilir
- Bölüm içi "Bölüm X Kaynakları" listeleri kaldırılır
- EK-A görsel galerisi kaldırılır; görseller ilgili bölümde gösterilir

Kullanım:
  python3 scripts/build_thesis_fbe.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
FIGS = REPO / "figures"
TEMPLATE = Path(os.path.expanduser("~/Downloads/yeni1- TEZ ŞABLONU.docx"))
PANDOC = shutil.which("pandoc") or os.path.expanduser("~/.local/bin/pandoc")
OUT = REPO / "Tez_GAN_Urban_Design_FBE.docx"
MASTER_MD = REPO / "build" / "thesis_master.md"
MASTER_MD.parent.mkdir(parents=True, exist_ok=True)


# Bölüm dosyaları sırası (FBE şablonu sırasında)
SECTIONS = [
    ("00_On_Sayfalar.md", "front"),
    ("04_Bolum1_Giris.md", "ch1"),
    ("03_Bolum2_Literatur.md", "ch2"),
    ("05_Bolum3_Yontem.md", "ch3"),
    ("06_Bolum4_Deneysel_Sonuclar.md", "ch4"),
    ("07_Bolum5_Tartisma.md", "ch5"),
    ("08_Bolum6_Sonuc.md", "ch6"),
]


# Bölüm başına eklenecek/görsele eklenecek satırlar
# Anahtar: bölüm dosyası
# Değer: [(marker_substring_for_anchor, image_markdown_block), ...]
FIGURE_INSERTS: dict[str, list[tuple[str, str]]] = {
    "05_Bolum3_Yontem.md": [
        # 3.1 sonunda Şekil 3.1
        ("Her iki aşama da **eşli (paired) görüntü çiftleri**",
         "\n\n![**Şekil 3.1.** Önerilen iki aşamalı kentsel peyzaj plan üretimi ve renklendirme pipeline'ı.](fig_3_1_pipeline.png){width=15cm}\n\n"),
        # 3.2.1 sonunda Şekil 4.1 (dataset örnekleri)
        ("`bash scripts/download_data.sh maps`",
         "\n\nŞekil 3.2'de `maps` veri kümesinden seçilmiş dört örnek üçlü gösterilmektedir: solda uydu (A) görüntüsü, ortada sentezlenmiş kroki, sağda Google Maps render'ı (hedef B). Veri seti Isola ve arkadaşları [4] tarafından Pix2Pix orijinal çalışmasında sunulduğu biçimde kullanılmıştır.\n\n![**Şekil 3.2.** `maps` veri kümesinden seçilmiş dört eşli üçlü örnek (uydu / sentetik kroki / renkli plan).](fig_4_1_dataset_examples.png){width=15cm}\n\n"),
        # 3.2.3 sonunda Şekil 4.2 (sketch yöntemleri)
        ("> **Repo yolu:** `scripts/prepare_sketches.py`, fonksiyon `make_sketch()`.",
         "\n\n![**Şekil 3.3.** Sentetik sketch üretiminin üç yöntemi: Canny kenar tespiti, XDoG filtresi ve ikisinin birleşimi (eşitlik 3.1–3.4).](fig_4_2_sketch_methods.png){width=15cm}\n\n"),
        # 3.3.1 sonunda Şekil 3.2 (U-Net) ve Şekil 3.3 (PatchGAN)
        ("> **Repo yolu:** `src/models/networks.py`",
         "\n\n![**Şekil 3.4.** U-Net jeneratör mimarisi: simetrik encoder-decoder yapısı ve sıçramalı bağlantılar [4].](fig_3_2_unet.png){width=15cm}\n\n![**Şekil 3.5.** PatchGAN ayırıcı mimarisi: 70×70 yamasal gerçeklik haritası üreten dört konvolüsyon katmanı [4].](fig_3_3_patchgan.png){width=14cm}\n\n"),
        # 3.3.4 sonunda Şekil 3.4 (SPADE)
        ("> **Repo yolu:** NVlabs/SPADE resmi implementasyon",
         "\n\n![**Şekil 3.6.** SPADE (uzamsal-uyarlamalı denormalizasyon) blok diyagramı [7]. Anlamsal etiket maskeden $(\\gamma,\\beta)$ uzamsal-uyarlamalı afin parametreler üretilir.](fig_3_4_spade.png){width=14cm}\n\n"),
    ],
    "06_Bolum4_Deneysel_Sonuclar.md": [
        # 4.3.3 perception-distortion sonunda Şekil 4.6
        ("**Tez açısından kritik sonuç:**",
         "\n\n![**Şekil 4.1.** Algı-bozulma ödünleşimi (Blau ve Michaeli, 2018 [33]) ve Geliştirilmiş Pix2Pix'in bu ödünleşimi aşması — FID × SSIM uzayında üç model konumu.](fig_4_6_tradeoff.png){width=13cm}\n\n"),
        # 4.4.3 sonunda Şekil 4.5 (iyileşme yüzdeleri)
        ("- **PSNR (+1,19 dB) ve L1 (−15,5 %)**",
         "\n\nGeliştirilmiş Pix2Pix'in baseline'a göre tüm beş metrikteki göreli iyileşme yüzdeleri Şekil 4.2'de sunulmuştur.\n\n![**Şekil 4.2.** Geliştirilmiş Pix2Pix'in Pix2Pix baseline'a göre beş metrik üzerinde göreli iyileşme yüzdeleri (pozitif değer = daha iyi).](fig_4_5_improvements.png){width=13cm}\n\n"),
        # 4.4 sonunda Şekil 4.8 (epoch progresi)
        ("Pix2PixHD'nin resmî implementasyonu çalıştırılamamış olsa da",
         "\n\nModelin eğitim sürecindeki ilerleyişi Şekil 4.3'te görselleştirilmiştir; epoch 0001'de henüz domain bilgisi öğrenilmemişken epoch 0076'da renk dağılımı oturmuş, epoch 0150'de ince detay seviyesinde hedefle örtüşme sağlanmıştır.\n\n![**Şekil 4.3.** Geliştirilmiş Pix2Pix modelinin val kümesindeki bir örnek için eğitim sürecindeki çıktı kalitesi gelişimi (epoch 0001 → 0076 → 0150).](fig_4_8_training_progress.png){width=14cm}\n\n"),
        # 4.6 başında Şekil 4.3 (bar) ve Şekil 4.4 (radar)
        ("**Tablo 4.5.** Üç GAN mimarisinin",
         "Karşılaştırmalı performansın bütüncül görünümü için Şekil 4.4 (sütun grafik) ve Şekil 4.5 (radar profil) sunulmuştur; Geliştirilmiş Pix2Pix beş metriğin tamamında altın çerçeveli (en iyi) konumdadır.\n\n![**Şekil 4.4.** Üç GAN mimarisinin beş değerlendirme metriğindeki performans karşılaştırması; altın çerçeveli sütunlar her metriğin en iyi sonucunu gösterir.](fig_4_3_metrics_bars.png){width=15cm}\n\n![**Şekil 4.5.** Modellerin normalize metrik profili (radar); dış kenar = daha iyi. Geliştirilmiş Pix2Pix yeşil profille beş eksende de en dışta.](fig_4_4_radar.png){width=12cm}\n\n"),
        # 4.6 niteliksel karşılaştırma — sıralama listesi sonrası
        ("1. **Geliştirilmiş Pix2Pix (512² + VGG)** — *önerilen ana yöntem*",
         "\n\nNiteliksel karşılaştırma val kümesinden seçilmiş dört örnek üzerinde Şekil 4.6'da sunulmuştur. Geliştirilmiş Pix2Pix sütunu, hedefe en yakın renk doygunluğu ve mekânsal doğruluğu sergilemektedir.\n\n![**Şekil 4.6.** Üç modelin niteliksel karşılaştırması: val kümesinden dört örnek için girdi kroki, gerçek hedef ve üç model çıktısı.](fig_4_7_qualitative.png){width=16cm}\n\n"),
    ],
}


# Bölüm sonu "Kaynaklar" / "Ek Kaynaklar" başlığını ve liste bloğunu kaldıracak desen
SECTION_BIB_PATTERN = re.compile(
    r"\n## Bölüm[^\n]*Kaynaklar[^\n]*\n.*?(?=\n## |\n# |\Z)",
    re.DOTALL,
)
# 1. seviye "Ek Kaynak" başlığı (Bölüm 5 sonunda)
EXTRA_BIB_PATTERN = re.compile(
    r"\n## Bölüm[^\n]*İçin Ek Kaynak[^\n]*\n.*?(?=\n## |\n# |\Z)",
    re.DOTALL,
)




def load_and_clean_section(path: Path, key: str) -> str:
    """Bir bölümü oku, bölüm sonu kaynakça bloklarını kaldır, görselleri yerleştir."""
    text = path.read_text(encoding="utf-8")

    # YAML front-matter kaldır
    if text.startswith("---"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :].lstrip()

    # "TEZ YAZIM REHBERİ" gibi build-only başlıkları kaldır
    text = re.sub(r"\n## TEZ YAZIM REHBERİ.*?(?=\n## |\n# |\Z)", "", text, flags=re.DOTALL)

    # İçindekiler / Şekiller / Tablolar listelerini placeholder olarak kaldır
    # (pandoc'un kendi TOC'si daha sonra eklenecek)
    text = re.sub(r"\n## İÇİNDEKİLER \(Şablon\).*?(?=\n## |\n# |\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\n## ŞEKİLLER LİSTESİ \(Şablon\).*?(?=\n## |\n# |\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\n## TABLOLAR LİSTESİ \(Şablon\).*?(?=\n## |\n# |\Z)", "", text, flags=re.DOTALL)

    # Bölüm sonu kaynakça bloklarını kaldır
    text = SECTION_BIB_PATTERN.sub("", text)
    text = EXTRA_BIB_PATTERN.sub("", text)

    # Görselleri uygun yerlere yerleştir
    for marker, fig_md in FIGURE_INSERTS.get(path.name, []):
        if marker in text:
            text = text.replace(marker, marker + fig_md, 1)
        else:
            print(f"  ! UYARI: '{marker[:50]}...' marker'ı {path.name} içinde bulunamadı; görsel atlandı")

    return text.strip() + "\n\n"


def collect_all_references() -> str:
    """Tüm bölüm sonu kaynakça bloklarını tek liste halinde topla, sıralı sun."""
    # Numerik atıfları [1] [2] ... ile bağlı olduğundan, bölüm 2'nin master listesi + ek kaynaklar
    # birleştirilir.
    parts = []

    # Bölüm 2 master kaynakça
    b2 = (DOCS / "03_Bolum2_Literatur.md").read_text(encoding="utf-8")
    m = re.search(r"## Bölüm 2 Kaynakları.*?(?=\n## |\Z)", b2, re.DOTALL)
    if m:
        block = m.group(0)
        # Başlığı kaldır, sadece [N] ile başlayan kayıtları al
        parts.append(re.sub(r"^## Bölüm 2.*\n", "", block).strip())

    # Bölüm 1 ek kaynaklar [29]-[32]
    b1 = (DOCS / "04_Bolum1_Giris.md").read_text(encoding="utf-8")
    m = re.search(r"## Bölüm 1 İçin Ek Kaynaklar.*?(?=\n## |\Z)", b1, re.DOTALL)
    if m:
        block = m.group(0)
        parts.append(re.sub(r"^## Bölüm 1.*\n", "", block).strip())

    # Bölüm 5 ek kaynak [33]
    b5 = (DOCS / "07_Bolum5_Tartisma.md").read_text(encoding="utf-8")
    m = re.search(r"## Bölüm 5 İçin Ek Kaynak.*?(?=\n## |\Z)", b5, re.DOTALL)
    if m:
        block = m.group(0)
        parts.append(re.sub(r"^## Bölüm 5.*\n", "", block).strip())

    full = "\n\n".join(parts).strip()

    # "Doğrulama notu" gibi meta yorumları kaldır
    full = re.sub(r"^> \*\*Doğrulama notu.*?(?=\n\*\*\[|\Z)", "", full, flags=re.DOTALL | re.MULTILINE)
    full = re.sub(r"^> \*\*Bilgi notu.*?(?=\n\*\*\[|\Z)", "", full, flags=re.DOTALL | re.MULTILINE)
    full = re.sub(r"^> .*$", "", full, flags=re.MULTILINE)

    # [N] referansları için Türkçe akademik atıf düzeni:
    # Mevcut format zaten "**[N]** Yazar, A. (yıl). Başlık. *Dergi*, ..."
    # Tek değişiklik: başlığı kaldırıp tek "KAYNAKLAR" altında topla
    return "\n\n# KAYNAKLAR\n\n" + full.strip() + "\n"


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"FBE şablonu yok: {TEMPLATE}")
    if not os.path.isfile(PANDOC):
        raise SystemExit(f"pandoc yok: {PANDOC}")

    # 1) Tüm bölümleri birleştir
    print(">> Master Markdown oluşturuluyor...")
    chunks = []
    # Pandoc metadata (Türkçe tarih biçimi vb.)
    chunks.append("---\ntitle: GAN Modelleri Kullanarak Kentsel Peyzaj Tasarım Planlarının Otomatik Oluşturulması ve Renklendirilmesi\nlang: tr-TR\n---\n\n")
    for filename, key in SECTIONS:
        src = DOCS / filename
        if not src.exists():
            print(f"  ! {src} bulunamadı, atlandı")
            continue
        print(f"  -> {filename}")
        chunks.append(load_and_clean_section(src, key))

    # 2) KAYNAKLAR (sona, tek liste)
    print(">> Kaynaklar birleştiriliyor...")
    chunks.append(collect_all_references())

    master = "".join(chunks)
    MASTER_MD.write_text(master, encoding="utf-8")
    print(f">> Master yazıldı: {MASTER_MD} ({len(master):,} byte)")

    # 3) Pandoc çağrısı
    print(">> Pandoc dönüşümü (şablon stilleri uygulanıyor)...")
    cmd = [
        PANDOC,
        "-f", "markdown+tex_math_dollars+raw_attribute+pipe_tables+table_captions+implicit_figures",
        "-t", "docx",
        str(MASTER_MD),
        "-o", str(OUT),
        "--reference-doc", str(TEMPLATE),
        "--resource-path", f"{FIGS}{os.pathsep}{FIGS / 'data'}{os.pathsep}{REPO}",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--metadata", "lang=tr-TR",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print("STDERR:\n", result.stderr[:2000])
        raise SystemExit(f"Pandoc başarısız (kod {result.returncode})")
    if result.stderr:
        print("Pandoc uyarıları:", result.stderr[:500])

    size_kb = OUT.stat().st_size / 1024
    print(f"\n>> Hazır: {OUT}  ({size_kb:.1f} KB)")

    # 4) Doğrulama
    print("\n>> Doğrulama:")
    import zipfile
    with zipfile.ZipFile(OUT) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
    omml = len(re.findall(r"<m:oMath[\s>/]", doc_xml))
    omml_para = len(re.findall(r"<m:oMathPara[\s>/]", doc_xml))
    images = len(re.findall(r"<w:drawing", doc_xml))
    tables = len(re.findall(r"<w:tbl[\s>]", doc_xml))
    print(f"   OMath toplam:      {omml}")
    print(f"   OMathPara display: {omml_para}")
    print(f"   Görseller:          {images}")
    print(f"   Tablolar:           {tables}")


if __name__ == "__main__":
    main()
