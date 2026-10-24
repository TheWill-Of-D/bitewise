# Optional custom Singapore food classifier

The default application uses a pretrained Food-101 model. This document describes the separate custom training path from the original BiteWise proposal; it has not been trained or evaluated on FoodSG-233 in this repository.

## Obtain and prepare data

FoodSG-233 access is subject to individual approval and the terms on its [official NUS page](https://foodlg.comp.nus.edu.sg/). Do not commit or redistribute the dataset or private access URLs. Select approximately 30 categories for the proposed experiment. Map category names explicitly to corresponding entries in `data/nutrition.csv`; add validated nutrition rows for categories missing from that table.

Prepare one directory per category:

```text
data/foodsg233_subset/
  Laksa/
    image_001.jpg
    image_002.jpg
    image_003.jpg
    ...
  Chicken rice (steamed)/
    image_001.jpg
    ...
```

At least three images per class are needed just to create splits. Use substantially more for a meaningful model evaluation. Audit duplicates before splitting so that near-identical images do not leak between subsets.

If using `scripts/download_foodsg.py`, supply the approved URL through `BITEWISE_DATASET_URL` and the provider's checksum through `--expected-md5`. The helper preserves a minimum 5 GiB disk reserve and monitors the physical Windows backing volume on WSL. It refuses to overwrite existing downloads. The application does not require this large dataset download.

## Obtain ImageNet initialization once

Install `requirements-model.txt` as described in the main README. Download the official torchvision EfficientNet-B0 initialization:

```bash
python - <<'PY'
from pathlib import Path
import torch
from torchvision.models import EfficientNet_B0_Weights
Path('models').mkdir(exist_ok=True)
torch.save(EfficientNet_B0_Weights.DEFAULT.get_state_dict(progress=True), 'models/imagenet_efficientnet_b0.pth')
PY
```

## Train and evaluate locally

```bash
python train_model.py \
  --data data/foodsg233_subset \
  --imagenet-weights models/imagenet_efficientnet_b0.pth
```

The script creates a seeded, class-stratified approximately 80/10/10 split, trains the classification head, fine-tunes the final EfficientNet blocks at a lower learning rate, and selects a checkpoint using validation top-1 accuracy with early stopping. Small classes reserve at least one image per split. Frozen blocks keep their BatchNorm statistics unchanged. Head-stage early stopping still allows the fine-tuning stage to run.

Optional independently collected photos use the same class-folder layout and can cover a subset of the trained classes:

```bash
python train_model.py \
  --data data/foodsg233_subset \
  --imagenet-weights models/imagenet_efficientnet_b0.pth \
  --external-test data/external_photos
```

The outputs under `models/` are `food_classifier.pt`, `class_names.json`, `test_metrics.json`, and `split_manifest.json`. The script reports top-1 accuracy, top-3 accuracy, and macro F1. Keep the test set out of hyperparameter selection. CPU training works but a GPU is more practical for a substantial dataset.

An explicit `--from-scratch` flag trains all layers without ImageNet initialization. This is useful for pipeline checks but does not follow the proposed transfer-learning baseline.

Restart the app with `BITEWISE_MODEL=custom` to use the checkpoint, or leave the default `auto` mode, which prioritizes installed custom weights. Measure CPU inference latency on your actual machine, test the external photos, validate nutrition calculations against reference records, and evaluate manual versus AI-assisted logging effort before reporting project results. Synthetic smoke-check results do not establish food-recognition quality.
