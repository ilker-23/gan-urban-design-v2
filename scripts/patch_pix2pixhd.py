"""NVIDIA pix2pixHD (2018) repo'sunu Python 3.9+ / PyTorch 2.x ile çalışır hale getirir.

Eski repo, modern Colab ortamında (Python 3.12) bir dizi uyumsuzluk içerir:
  1) train.py:  `fractions.gcd`  -> Python 3.9'da kaldırıldı; `math.gcd` kullan.
  2) train.py:  `.data[0]`        -> PyTorch 0.4+'da skaler tensör indekslenemez; `.item()`.
  3) models/base_model.py: `raise('...')` (Python 2 stili) -> `raise RuntimeError('...')`.

Idempotent: birden fazla çalıştırma güvenlidir; sadece eşleşen kalıpları değiştirir.

Kullanım:
  python scripts/patch_pix2pixhd.py --repo external/pix2pixHD
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str, str]]) -> int:
    """replacements: (regex_pattern, replacement, açıklama). Değişen kalıp sayısını döndürür."""
    if not path.exists():
        print(f"  ATLA (yok): {path}")
        return 0
    text = path.read_text(encoding="utf-8")
    n_total = 0
    for pattern, repl, desc in replacements:
        new_text, n = re.subn(pattern, repl, text)
        if n:
            print(f"  ✓ {path.name}: {desc} ({n} yer)")
            text = new_text
            n_total += n
    if n_total:
        path.write_text(text, encoding="utf-8")
    return n_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="external/pix2pixHD", type=Path)
    args = ap.parse_args()

    repo = args.repo
    if not repo.exists():
        raise SystemExit(f"pix2pixHD repo yok: {repo} — önce setup_external.sh çalıştırın.")

    print(f">> pix2pixHD yamalanıyor: {repo}")

    total = 0

    # 1) train.py — fractions.gcd + .data[0]
    total += patch_file(repo / "train.py", [
        (r"import fractions",
         "import fractions\nimport math",
         "math import eklendi"),
        (r"fractions\.gcd",
         "math.gcd",
         "fractions.gcd -> math.gcd"),
        (r"abs\(a \* b\)\s*/\s*math\.gcd",
         "abs(a * b) // math.gcd",
         "lcm tam sayı bölmesi"),
        (r"\.data\[0\]",
         ".item()",
         ".data[0] -> .item() (skaler loss)"),
    ])

    # 2) base_model.py — raise('string') Python 2 stili
    total += patch_file(repo / "models" / "base_model.py", [
        (r"raise\(\s*('[^']*')\s*\)",
         r"raise RuntimeError(\1)",
         "raise('...') -> raise RuntimeError('...')"),
        (r'raise\(\s*("[^"]*")\s*\)',
         r"raise RuntimeError(\1)",
         'raise("...") -> raise RuntimeError("...")'),
    ])

    # 3) util/util.py — bazı sürümlerde scipy.misc kullanımı olabilir (varsa uyar)
    util_path = repo / "util" / "util.py"
    if util_path.exists() and "scipy.misc" in util_path.read_text(encoding="utf-8"):
        print("  ! UYARI: util/util.py içinde scipy.misc kullanımı var; "
              "test sırasında hata verirse manuel imageio'ya çevrilmeli.")

    if total == 0:
        print(">> Değişiklik yok (zaten yamalı olabilir).")
    else:
        print(f">> Bitti: toplam {total} kalıp yamalandı.")


if __name__ == "__main__":
    main()
