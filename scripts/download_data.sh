#!/usr/bin/env bash
# Hazır datasetleri indirir.
# Kullanım:
#   bash scripts/download_data.sh maps        # Berkeley pix2pix maps dataset
#   bash scripts/download_data.sh deepglobe   # Kaggle DeepGlobe Land Cover
#   bash scripts/download_data.sh all
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

download_maps() {
  echo ">> Maps (Berkeley Pix2Pix) dataset indiriliyor..."
  cd "$DATA_DIR"
  if [ -d "maps" ]; then
    echo "   data/maps zaten var, atlandı."
  else
    wget -nc http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
    tar -xzf maps.tar.gz
    rm -f maps.tar.gz
    echo "   -> data/maps/{train,val} hazır."
  fi
  cd - >/dev/null
}

download_deepglobe() {
  echo ">> DeepGlobe Land Cover (Kaggle) dataset indiriliyor..."
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "HATA: kaggle CLI yok. 'pip install kaggle' ve ~/.kaggle/kaggle.json yapılandırması gerekiyor."
    exit 1
  fi
  mkdir -p "$DATA_DIR/deepglobe"
  cd "$DATA_DIR/deepglobe"
  if compgen -G "*.jpg" > /dev/null || compgen -G "train/*.jpg" > /dev/null; then
    echo "   DeepGlobe zaten var, atlandı."
  else
    kaggle datasets download -d balraj98/deepglobe-land-cover-classification-dataset
    unzip -q deepglobe-land-cover-classification-dataset.zip
    rm -f deepglobe-land-cover-classification-dataset.zip
    echo "   -> data/deepglobe/ hazır."
  fi
  cd - >/dev/null
}

case "${1:-all}" in
  maps)       download_maps ;;
  deepglobe)  download_deepglobe ;;
  all)        download_maps; download_deepglobe ;;
  *)          echo "Bilinmeyen seçenek: $1. Geçerli: maps | deepglobe | all"; exit 1 ;;
esac

echo "Bitti."
