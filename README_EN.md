# Part-Aware Hyena Individual Identification

**English** | [简体中文](README.md)

The 4th-place solution to the iFLYTEK Hyena Individual Identification Challenge
（野生鬣狗个体识别挑战赛）, with an online Macro-F1 of **0.77415**. The method
combines part-aware representations, heterogeneous visual backbones, label-free
foreground observations, and part-wise raw-score ensembling under a strict
source-group validation protocol.

This repository provides data restoration, training, inference, ensembling,
leakage audits, and exact reconstruction of the released submission.

## Method Overview

- **Part-aware backbone**: DINOv3-H+/16 models jointly learn global, stripe, and
  covariance descriptors, with complementary representations for head and body views.
- **Heterogeneous models**: DINOv3 ConvNeXt-L contributes convolutional inductive
  bias, while BioCLIP ViT-B/16 provides a distinct pretraining representation space.
- **Foreground observation**: a frozen SAM2.1 Hiera-Tiny model generates label-free
  foreground masks through the same deterministic pipeline for train and test images.
- **Cross-domain transfer**: public DogFaceNet dog-identity data is used for generic
  individual representation learning. Its external identity head is discarded, and
  the 255 competition classifiers are initialized from scratch.
- **Part-wise ensemble**: raw scores for `head`, `left_body`, and `right_body` are
  combined separately. Every base model uses one non-TTA forward pass.

See the [method description](docs/METHOD.md) for architectures, objectives, and
ensemble equations. The machine-readable recipe is available in
[configs/final_pipeline.yaml](configs/final_pipeline.yaml).

## Repository Layout

```text
.
├── configs/        # final pipeline configuration
├── src/hyenaid/    # models, data, losses, and training engine
├── tools/          # training, inference, ensembling, and audit tools
├── scripts/        # end-to-end entry points
├── benchmark/      # fixed folds, metrics, and leakage checks
├── artifacts/      # frozen raw scores and official submission records
└── docs/METHOD.md  # method description
```

## Reproduction

### 1. Environment

Reference environment: Linux, Python 3.11, CUDA 12.8, PyTorch 2.9.1, and one
RTX 3090 24 GB GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make verify
```

For a different CUDA runtime, install matching `torch` and `torchvision` wheels
from the official PyTorch index before installing the remaining dependencies.

### 2. Data Preparation

Download the public competition bundle from the iFLYTEK platform and the Hyena ID
2022 image archive referenced by the official package:

- [iFLYTEK Challenge Platform](https://challenge.xfyun.cn/)
- [LILA Hyena ID 2022](https://lila.science/datasets/hyena-id-2022/)
- [Kaggle mirror](https://www.kaggle.com/datasets/ashfaqsyed/hyena-dataset)

```bash
make prepare-data \
  PUBLIC_DATA_ZIP=/path/to/dataset\(public\).zip \
  HYENA_DATA_ZIP=/path/to/hyena-dataset.zip
```

The script restores only the 4,067 official training crops and verifies the fixed
source-group folds. No additional hyena images from the archive are added to training.

### 3. Weights

The 7.8 GB bundle contains the public initializers, SAM2 checkpoint, and DogFace
representation checkpoint required for full training:

> [Baidu Netdisk: download the complete `weights` directory](https://pan.baidu.com/s/15SQ2cEHe5u3Z0TI1WKxdUQ?pwd=gv24)  
> Extraction code: `gv24`

Keep all filenames unchanged and place the directory next to the repository:

```text
workspace/
├── IFLYTEK-HyenaDetection2026-4th-Solution/
└── weights/
```

```bash
make verify-weights
```

Set `WEIGHTS_ROOT=/absolute/path/to/weights` when using another location. Source
URLs, pinned revisions, licenses, and SHA-256 checksums are recorded in
`weights/SOURCE_MANIFEST.json` and `weights/SHA256SUMS`.

### 4. Exact Released-Submission Reconstruction

The repository includes the 255-class raw scores from the official run. The released
submission can therefore be rebuilt without a GPU or source images:

```bash
make reproduce
sha256sum work/frozen_rebuild/submission.csv
```

Expected CSV SHA-256:

```text
a37db3534c9476228799dd5acc8ee749cd21040980235f2fc2880820d632d678
```

This reproduces the released predictions exactly; it is not a fresh model forward
pass from final competition checkpoints.

### 5. Direct Inference from Final Checkpoints

The current public weight bundle does not contain the eight final competition
checkpoints. This release therefore **cannot** skip training and run fresh final-model
inference on new images. Use the full pipeline below for model-level predictions;
the frozen raw scores above are provided only to verify the released submission.

### 6. Full Training and Inference

Generate the deterministic foreground masks:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
export SAM2_REPO=$PWD/third_party/sam2
make masks
```

Then train all eight models, run one fixed non-TTA test forward pass per model,
and perform the part-wise ensemble:

```bash
make train
```

Outputs:

```text
work/full_run/final_submission/submission.csv
work/full_run/final_submission/submission.zip
work/full_run/final_submission/reproduction_report.json
```

A historical full run took approximately 25 hours on one RTX 3090. Epoch counts,
learning rates, loss weights, and execution order are fixed in the configuration
and entry-point scripts.

## Data and Compliance

- Official training set: 4,067 crops and 255 identities. The fixed fold-manifest
  SHA-256 is `0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f`.
- DogFaceNet data comes from [Zenodo 12578449](https://zenodo.org/records/12578449)
  under CC BY 4.0: 8,363 images and 1,393 dog identities. Its classifier is discarded.
- Decoded-pixel SHA-256 auditing found no exact overlap between DogFace data and
  official train or anonymous test images.
- No extra HyenaID identities, same-source animal-ReID weights, test labels,
  pseudo-labels, anonymous-test identity recovery, or competition-checkpoint warm starts.
- SAM2 performs label-free foreground segmentation only. Anonymous test images never
  enter a supervised training step.

DogFace pretraining is used as generic individual representation learning across a
different species and identity space. Its provenance, license, and transfer boundary
are fully disclosed. Final rule interpretation remains with the competition organizer.

## Acknowledgements

This project uses or builds upon [timm](https://github.com/huggingface/pytorch-image-models),
[DINOv3](https://github.com/facebookresearch/dinov3),
[SAM2](https://github.com/facebookresearch/sam2),
[BioCLIP](https://github.com/Imageomics/bioclip), and public DogFaceNet resources.
Third-party assets remain subject to their original licenses.

## License

Original code in this repository is released under the [MIT License](LICENSE).
Competition data, external datasets, pretrained weights, and third-party code are
not covered by this license and remain subject to their original terms.
