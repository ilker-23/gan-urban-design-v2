"""Tezin statik (veriye-bağlı olmayan) görsellerini üretir:
  Şekil 3.1 — İki aşamalı pipeline diyagramı
  Şekil 3.2 — U-Net jeneratör mimarisi
  Şekil 3.3 — PatchGAN ayırıcı mimarisi
  Şekil 3.4 — SPADE blok diyagramı
  Şekil 4.3 — 3-model × 5-metrik karşılaştırma bar chart (normalize)
  Şekil 4.4 — 3-model radar chart
  Şekil 4.5 — Geliştirilmiş Pix2Pix vs baseline iyileşme yüzdeleri
  Şekil 4.6 — Algı-bozulma ödünleşimi scatter (FID vs SSIM)

Çıktı: figures/*.png (300 DPI, tezde yayımlanabilir kalitede)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

# Akademik palette
COL_INPUT = "#4C72B0"     # mavi
COL_OUTPUT = "#55A868"    # yeşil
COL_MIDDLE = "#C44E52"    # kırmızımsı
COL_GAN_PIX = "#4C72B0"
COL_GAN_CYC = "#DD8452"
COL_GAN_ENH = "#55A868"
COL_BG = "#F5F5F5"


# =============================================================================
# Şekil 3.1 — İki aşamalı pipeline diyagramı
# =============================================================================
def fig_3_1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=10):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05,rounding_size=0.15",
                           facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=20,
                                     linewidth=2, color="black"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, label,
                    ha="center", fontsize=9, style="italic")

    # Sol — Stage 1
    box(0.2, 2.0, 2.0, 1.4, "Uydu /\nAerial", COL_INPUT)
    box(3.0, 1.8, 3.0, 1.8, "Stage-1:\nPlan Üretimi\n(Pix2PixHD-benzeri)", COL_MIDDLE)
    box(6.8, 2.0, 2.0, 1.4, "Semantic\nMap", COL_OUTPUT, fontsize=11)

    arrow(2.2, 2.7, 2.95, 2.7)
    arrow(6.05, 2.7, 6.8, 2.7)

    # Sağ — Stage 2
    box(0.2, 0.1, 2.0, 1.4, "Sentetik\nKroki", COL_INPUT)
    box(3.0, -0.1, 3.0, 1.8, "Stage-2:\nRenklendirme\n(Pix2Pix / CycleGAN /\nEnhanced)", COL_MIDDLE, fontsize=9)
    box(6.8, 0.1, 2.0, 1.4, "Renkli\nPlan", COL_OUTPUT, fontsize=11)

    arrow(2.2, 0.8, 2.95, 0.8)
    arrow(6.05, 0.8, 6.8, 0.8)

    # Stage-1 -> Stage-2 (semantic map birleşir)
    ax.add_patch(FancyArrowPatch((7.8, 2.0), (7.8, 1.5),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color="gray",
                                 linestyle="--"))
    ax.text(8.0, 1.7, "ürün", ha="left", fontsize=8, style="italic", color="gray")

    # Çıktı kullanım kutusu
    box(9.5, 1.0, 2.2, 2.4,
        "Kentsel Plan\nGörselleştirme\n+\nDoküman",
        "#3F3F3F", fontsize=10)
    arrow(8.8, 2.7, 9.5, 2.7)
    arrow(8.8, 0.8, 9.5, 0.8)

    # Başlık
    ax.text(6, 4.5, "Şekil 3.1 — Önerilen İki Aşamalı GAN Pipeline'ı",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6, 4.1, "Stage-1: Plan üretimi (aerial → semantic map)  •  Stage-2: Sketch renklendirme (kroki → renkli plan)",
            ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(OUT / "fig_3_1_pipeline.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_3_1_pipeline.png'}")


# =============================================================================
# Şekil 3.2 — U-Net Generator
# =============================================================================
def fig_3_2_unet():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    enc_dims = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    chans    = [3,   64,  128, 256, 512, 512, 512, 512, 512]

    # Encoder (sol yarı, aşağı doğru)
    n = len(enc_dims)
    cx_enc = np.linspace(0.7, 5.5, n)
    enc_top_y = 4.0  # üst etiket düzeyi (encoder başlığı için)
    for i, (dim, ch) in enumerate(zip(enc_dims, chans)):
        h = max(0.4, math.log2(dim + 1) * 0.35)
        w = 0.5
        y_center = 4.0
        rect = Rectangle((cx_enc[i] - w/2, y_center - h/2), w, h,
                         facecolor=COL_INPUT, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx_enc[i], y_center - h/2 - 0.2, f"{dim}²", ha="center", fontsize=7)
        ax.text(cx_enc[i], y_center + h/2 + 0.05, f"{ch}", ha="center", fontsize=7, color="#333")

    # Decoder (sağ yarı)
    dec_dims = enc_dims[::-1]
    dec_chans = chans[::-1]
    cx_dec = np.linspace(6.0, 11.0, n)
    for i, (dim, ch) in enumerate(zip(dec_dims, dec_chans)):
        h = max(0.4, math.log2(dim + 1) * 0.35)
        w = 0.5
        y_center = 4.0
        rect = Rectangle((cx_dec[i] - w/2, y_center - h/2), w, h,
                         facecolor=COL_OUTPUT, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx_dec[i], y_center - h/2 - 0.2, f"{dim}²", ha="center", fontsize=7)
        ax.text(cx_dec[i], y_center + h/2 + 0.1, f"{ch}", ha="center", fontsize=7, color="#333")

    # Skip connections
    for i in range(n - 1):
        x1 = cx_enc[i]; x2 = cx_dec[n - 1 - i]
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x1, 4.5), (x2, 4.5),
                                         connectionstyle=f"arc3,rad={-0.2 - 0.04*i}",
                                         arrowstyle="-", linewidth=0.7,
                                         color="#888", linestyle=(0, (3, 2))))

    # Input/output etiketleri
    ax.annotate("Girdi\n(sketch)", (cx_enc[0], 3.0), ha="center",
                fontsize=9, fontweight="bold", color=COL_INPUT)
    ax.annotate("Çıktı\n(renkli plan)", (cx_dec[-1], 3.0), ha="center",
                fontsize=9, fontweight="bold", color=COL_OUTPUT)

    # Aşağı/yukarı oklar — başlıklar daha yukarıya
    ax.text(3, 5.55, "Encoder (downsample 4×4 conv s=2 + InstanceNorm + LeakyReLU)",
            ha="center", fontsize=8.5, style="italic", color="#333")
    ax.text(8.5, 5.55, "Decoder (upsample 4×4 transpose conv s=2 + InstanceNorm + ReLU)",
            ha="center", fontsize=8.5, style="italic", color="#333")
    ax.text(6, 5.25, "skip connections (kesik çizgi)", ha="center",
            fontsize=8, color="#666", style="italic")

    # Başlık
    ax.text(6, 5.85, "Şekil 3.2 — U-Net Jeneratör Mimarisi (Pix2Pix / Geliştirilmiş Pix2Pix)",
            ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 6.2)
    ax.text(6, 0.5,
            "Bloklar üzerindeki sayılar kanal sayısını (üst) ve uzamsal boyutu (alt) gösterir.\n"
            "256² girdi için num_downs=8;  512² Geliştirilmiş Pix2Pix için num_downs=9.",
            ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    plt.savefig(OUT / "fig_3_2_unet.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_3_2_unet.png'}")


# =============================================================================
# Şekil 3.3 — PatchGAN ayırıcı
# =============================================================================
def fig_3_3_patchgan():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    # Girdi: A (sketch) + B (real or fake), kanal-yığın
    ax.add_patch(Rectangle((0.3, 3.0), 1.0, 1.0, facecolor=COL_INPUT,
                           edgecolor="black", alpha=0.85))
    ax.text(0.8, 3.5, "A\n(sketch)", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
    ax.add_patch(Rectangle((0.3, 1.7), 1.0, 1.0, facecolor=COL_OUTPUT,
                           edgecolor="black", alpha=0.85))
    ax.text(0.8, 2.2, "B\n(real/fake)", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Concat
    ax.add_patch(FancyArrowPatch((1.3, 3.5), (2.0, 3.0),
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.5))
    ax.add_patch(FancyArrowPatch((1.3, 2.2), (2.0, 3.0),
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.5))
    ax.add_patch(FancyBboxPatch((2.0, 2.7), 0.7, 0.7,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor="#3F3F3F", edgecolor="black"))
    ax.text(2.35, 3.05, "Concat\n(6 ch)", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

    # Conv katmanları
    layers = [(3.2, 256, 64), (4.4, 128, 128), (5.6, 64, 256), (6.8, 32, 512), (8.0, 30, 1)]
    for x, dim, ch in layers:
        h = 1.5 if ch < 512 else 1.6
        rect = Rectangle((x, 2.4), 0.5, h * 0.6,
                         facecolor=COL_MIDDLE if ch != 1 else "#9B59B6",
                         edgecolor="black", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + 0.25, 2.4 + h*0.6 + 0.15, f"{ch}", ha="center", fontsize=8)
        ax.text(x + 0.25, 2.3, f"{dim}²", ha="center", fontsize=8, color="#444")

    # Conv ok'lar
    for i in range(len(layers) - 1):
        x1 = layers[i][0] + 0.5; x2 = layers[i+1][0]
        ax.add_patch(FancyArrowPatch((x1, 3.05), (x2, 3.05),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.2))

    # Patch çıktı haritası
    ax.text(8.25, 2.0, "30×30\nyamasal\ngerçeklik\nharitası",
            ha="center", fontsize=8.5, style="italic", color="#9B59B6", fontweight="bold")

    # Etiketler
    ax.text(5.5, 4.4, "70×70 PatchGAN: Her piksel, girdi görüntüsündeki bir 70×70 yamayı temsil eder.",
            ha="center", fontsize=9, style="italic", color="#333")

    # Başlık
    ax.text(5.5, 4.8, "Şekil 3.3 — PatchGAN Ayırıcı (Discriminator) Mimarisi",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.5, 0.7,
            "4 evrişim katmanı: ilk üçü stride=2, son ikisi stride=1; her katmanda 4×4 conv + InstanceNorm + LeakyReLU(0.2).\n"
            "Çıktı sigmoid uygulanmadan döndürülür (LSGAN için MSE doğrudan hesaplanır).",
            ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    plt.savefig(OUT / "fig_3_3_patchgan.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_3_3_patchgan.png'}")


# =============================================================================
# Şekil 3.4 — SPADE Blok
# =============================================================================
def fig_3_4_spade():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=9, fc=None):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")

    # Aktivasyon h_i
    box(0.5, 2.5, 1.2, 0.9, "$h^i$\n(aktivasyon)", COL_INPUT, fontsize=8)

    # BatchNorm
    box(2.3, 2.5, 1.2, 0.9, "BatchNorm\n($\\mu, \\sigma$)", "#666", fontsize=8)

    # SPADE element-wise affine
    box(4.0, 2.5, 1.6, 0.9, "$\\gamma \\cdot \\hat h + \\beta$\n(uzamsal-uyarlamalı)", COL_MIDDLE, fontsize=8)

    # Çıktı h'
    box(6.2, 2.5, 1.0, 0.9, "$h'^i$", COL_OUTPUT, fontsize=10)

    # Semantic mask + conv -> gamma, beta (üstte)
    box(2.3, 4.0, 1.7, 0.7, "Semantic Mask $\\mathbf{m}$", "#9B59B6", fontsize=8)
    box(4.5, 4.0, 1.3, 0.7, "Conv (shared)", "#9B59B6", fontsize=8)
    box(6.2, 4.5, 0.6, 0.5, "$\\gamma$", COL_MIDDLE, fontsize=10)
    box(6.2, 3.7, 0.6, 0.5, "$\\beta$", COL_MIDDLE, fontsize=10)

    # Oklar
    arrows = [
        ((1.7, 2.95), (2.3, 2.95)),
        ((3.5, 2.95), (4.0, 2.95)),
        ((5.6, 2.95), (6.2, 2.95)),
        ((4.0, 4.35), (4.5, 4.35)),
        ((5.8, 4.35), (6.2, 4.75)),
        ((5.8, 4.35), (6.2, 3.95)),
        ((6.8, 4.5), (6.0, 3.2)),    # gamma -> spade box
        ((6.8, 3.9), (6.0, 3.0)),    # beta -> spade box
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.2))

    # Denklem
    ax.text(5, 1.5,
            r"$\mathrm{SPADE}(h^i_{n,c,y,x}\mid\mathbf{m}) = "
            r"\gamma^i_{c,y,x}(\mathbf{m}) \cdot \frac{h^i_{n,c,y,x} - \mu^i_c}{\sigma^i_c} + \beta^i_{c,y,x}(\mathbf{m})$",
            ha="center", fontsize=11, color="#222")

    # Başlık
    ax.text(5, 4.85, "Şekil 3.4 — SPADE (Uzamsal-Uyarlamalı Denormalizasyon) Blok Diyagramı",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 0.7,
            "Anlamsal etiket $\\mathbf{m}$'den öğrenilen $(\\gamma, \\beta)$ uzamsal-uyarlamalı afin parametreler,\n"
            "BatchNorm sonrası aktivasyona piksel-bazında uygulanır; bu sayede etiket bilgisi derin katmanlarda 'yıkanmaz'.",
            ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    plt.savefig(OUT / "fig_3_4_spade.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_3_4_spade.png'}")


# =============================================================================
# Veriler: 3 modelin metrikleri
# =============================================================================
MODELS = ["Pix2Pix\n(256, L1)", "CycleGAN\n(256, eşsiz)", "Geliştirilmiş\nPix2Pix\n(512, L1+VGG)"]
COLORS = [COL_GAN_PIX, COL_GAN_CYC, COL_GAN_ENH]
METRICS = {
    "FID ↓":   [151.84, 54.58, 45.16],
    "SSIM ↑":  [0.7772, 0.6578, 0.8295],
    "PSNR ↑":  [27.36, 24.01, 28.55],
    "LPIPS ↓": [0.1766, 0.2032, 0.1641],
    "L1 ↓":    [8.01, 11.72, 6.77],
}


# =============================================================================
# Şekil 4.3 — 3-model × 5-metrik grouped bar chart (normalize)
# =============================================================================
def fig_4_3_metrics_bars():
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.5))
    metric_names = list(METRICS.keys())
    for i, mname in enumerate(metric_names):
        ax = axes[i]
        vals = METRICS[mname]
        bars = ax.bar(range(3), vals, color=COLORS, edgecolor="black", linewidth=1.0)
        # Etiketler
        for j, (b, v) in enumerate(zip(bars, vals)):
            offset = (max(vals) - min(vals)) * 0.05
            ax.text(b.get_x() + b.get_width()/2, v + offset,
                    f"{v:.2f}" if v >= 1 else f"{v:.4f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Pix2Pix", "CycleGAN", "Enhanced"], fontsize=8, rotation=0)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        # y-eksen başlığı sadece ilkinde
        if i == 0:
            ax.set_ylabel("Değer", fontsize=10)
        # En iyi bar'a yıldız (yön bilgisine göre)
        is_lower_better = "↓" in mname
        best_idx = int(np.argmin(vals)) if is_lower_better else int(np.argmax(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    fig.suptitle("Şekil 4.3 — Üç GAN Mimarisinin Beş Metrik Üzerinde Performans Karşılaştırması",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Altın çerçeveli sütunlar her metriğin en iyi sonucunu gösterir. "
             "Geliştirilmiş Pix2Pix (512²+VGG) beş metriğin tamamında üstündür.",
             ha="center", fontsize=9, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(OUT / "fig_4_3_metrics_bars.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_4_3_metrics_bars.png'}")


# =============================================================================
# Şekil 4.4 — Radar chart (normalize: tüm metrikler 0..1, yön düzeltilmiş)
# =============================================================================
def fig_4_4_radar():
    """Tüm metrikleri 0..1 aralığına normalize et; 'düşük iyi' metriklerini ters çevir
    böylece her eksende dış kenar = daha iyi."""
    keys = ["FID", "SSIM", "PSNR", "LPIPS", "L1"]
    lower_is_better = {"FID": True, "SSIM": False, "PSNR": False, "LPIPS": True, "L1": True}
    raw = {
        "Pix2Pix":   [151.84, 0.7772, 27.36, 0.1766, 8.01],
        "CycleGAN":  [54.58, 0.6578, 24.01, 0.2032, 11.72],
        "Enhanced":  [45.16, 0.8295, 28.55, 0.1641, 6.77],
    }
    # Min-max normalize + ters çevirme
    normed = {m: [] for m in raw}
    for i, k in enumerate(keys):
        vals = [raw[m][i] for m in raw]
        lo, hi = min(vals), max(vals)
        for m in raw:
            v = raw[m][i]
            n = (v - lo) / (hi - lo + 1e-9)
            if lower_is_better[k]:
                n = 1 - n
            normed[m].append(n)

    angles = np.linspace(0, 2*np.pi, len(keys), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    color_map = {"Pix2Pix": COL_GAN_PIX, "CycleGAN": COL_GAN_CYC, "Enhanced": COL_GAN_ENH}

    for m, vals in normed.items():
        v = vals + vals[:1]
        ax.plot(angles_closed, v, "o-", linewidth=2, label=m, color=color_map[m])
        ax.fill(angles_closed, v, alpha=0.15, color=color_map[m])

    ax.set_xticks(angles)
    ax.set_xticklabels(keys, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0 (en iyi)"], fontsize=8, color="#666")
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10)
    ax.set_title("Şekil 4.4 — Modellerin Normalize Metrik Profili (dış = daha iyi)",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUT / "fig_4_4_radar.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_4_4_radar.png'}")


# =============================================================================
# Şekil 4.5 — Enhanced vs baseline iyileşme yüzdeleri
# =============================================================================
def fig_4_5_improvements():
    metric_names = ["FID", "SSIM", "PSNR", "LPIPS", "L1"]
    base = np.array([151.84, 0.7772, 27.36, 0.1766, 8.01])
    enh  = np.array([45.16,  0.8295, 28.55, 0.1641, 6.77])
    # İyileşme yüzdesi (her metriğin yönüne göre işareti pozitif "iyileşme"ye ayarla)
    raw_pct = (enh - base) / base * 100
    improvement = [-raw_pct[0], raw_pct[1], raw_pct[2], -raw_pct[3], -raw_pct[4]]
    # "↓ metriklerinde düşüş = iyileşme" işaretle pozitif

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(metric_names, improvement, color=COL_GAN_ENH, edgecolor="black", linewidth=1.0)
    for b, v in zip(bars, improvement):
        ax.text(v + (1.5 if v > 0 else -1.5), b.get_y() + b.get_height()/2,
                f"{v:+.1f} %", va="center",
                ha="left" if v > 0 else "right",
                fontsize=11, fontweight="bold",
                color="#2C7A2C" if v > 0 else "#B33A3A")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("İyileşme yüzdesi (pozitif = baseline'a göre daha iyi)", fontsize=10)
    ax.set_title("Şekil 4.5 — Geliştirilmiş Pix2Pix'in Baseline'a Göre Metrik Bazlı İyileşmesi",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT / "fig_4_5_improvements.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_4_5_improvements.png'}")


# =============================================================================
# Şekil 4.6 — Algı-bozulma ödünleşimi: FID vs SSIM scatter
# =============================================================================
def fig_4_6_tradeoff():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    models = ["Pix2Pix (256, L1)", "CycleGAN (256, eşsiz)", "Geliştirilmiş Pix2Pix (512, L1+VGG)"]
    fids   = [151.84, 54.58, 45.16]
    ssims  = [0.7772, 0.6578, 0.8295]
    markers = ["o", "s", "*"]
    sizes   = [220, 220, 600]
    colors  = [COL_GAN_PIX, COL_GAN_CYC, COL_GAN_ENH]

    for m, f, s, mk, sz, c in zip(models, fids, ssims, markers, sizes, colors):
        ax.scatter(f, s, s=sz, c=c, marker=mk, edgecolors="black", linewidths=1.5,
                   zorder=3, label=m)

    # Etiketler (Enhanced için sağa kaydır — yıldız+ok çakışmasını önle)
    label_offsets = [(8, 0.005), (-3, -0.020), (10, -0.012)]
    for m, f, s, (dx, dy) in zip(models, fids, ssims, label_offsets):
        ax.annotate(m.split("(")[0].strip(),
                    xy=(f, s), xytext=(f + dx, s + dy),
                    fontsize=10, fontweight="bold")

    # "Daha iyi" oku — sağ alttan sol üste, yıldıza dokunmayacak şekilde
    ax.annotate("", xy=(28, 0.870), xytext=(120, 0.660),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=2))
    ax.text(75, 0.760, "daha iyi yönü\n(↓ FID, ↑ SSIM)",
            ha="center", fontsize=10, color="#666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    # Ödünleşim çizgisi (Pix2Pix - CycleGAN arası)
    ax.plot([fids[0], fids[1]], [ssims[0], ssims[1]],
            "--", color="#999", linewidth=1.5, zorder=1, alpha=0.5)
    ax.text(140, 0.715, "klasik\nalgı-bozulma\nödünleşimi",
            fontsize=9, color="#999", style="italic", ha="center")

    ax.set_xlabel("FID (↓ daha iyi)", fontsize=11, fontweight="bold")
    ax.set_ylabel("SSIM (↑ daha iyi)", fontsize=11, fontweight="bold")
    ax.set_title("Şekil 4.6 — Algı-Bozulma Ödünleşimi ve Geliştirilmiş Pix2Pix'in Aşması",
                 fontsize=12, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(20, 170)
    ax.set_ylim(0.62, 0.88)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT / "fig_4_6_tradeoff.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT/'fig_4_6_tradeoff.png'}")


# =============================================================================
# Main
# =============================================================================
def main():
    print(f">> Statik tez görselleri üretiliyor: {OUT}")
    fig_3_1_pipeline()
    fig_3_2_unet()
    fig_3_3_patchgan()
    fig_3_4_spade()
    fig_4_3_metrics_bars()
    fig_4_4_radar()
    fig_4_5_improvements()
    fig_4_6_tradeoff()
    print(f">> Bitti. {len(list(OUT.glob('*.png')))} PNG üretildi.")


if __name__ == "__main__":
    main()
