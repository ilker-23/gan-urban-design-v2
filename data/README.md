# Veri Dizini

Bu klasör **gitignore** altındadır; veri buraya indirilir ama repo'ya işlenmez.

## İndirme

```bash
# maps dataset (birincil) - ~250 MB
bash scripts/download_data.sh maps

# DeepGlobe (ikincil, SPADE için) - Kaggle hesabı gerekli
bash scripts/download_data.sh deepglobe
```

## Beklenen Yapı

```
data/
├── maps/
│   ├── train/         # 1096 × .jpg, her biri 1200×600 (yan yana: sat | map)
│   └── val/           # 1098 × .jpg
├── deepglobe/
│   ├── train/         # *_sat.jpg + *_mask.png
│   └── ...
└── processed/
    ├── maps_sketch/
    │   ├── train/
    │   │   ├── A/     # sentetik sketch (B&W)
    │   │   └── B/     # renkli map (target)
    │   └── val/
    └── maps_paired/
        ├── train/
        │   ├── A/     # satellite
        │   └── B/     # map
        └── val/
```

## Önişleme

```bash
python scripts/prepare_sketches.py --input data/maps --output data/processed/maps_sketch
```

Bu komut:
1. Her `1200×600` görseli ortadan 2 parçaya ayırır → `A=sat`, `B=map`.
2. Map'ten Canny + XDoG ile sentetik sketch üretir → `sketch`.
3. `(sketch, map)` çiftlerini `data/processed/maps_sketch/` altına yazar.
