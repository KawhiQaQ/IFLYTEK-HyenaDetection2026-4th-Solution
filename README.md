# Part-Aware Hyena Individual Identification

[English](README_EN.md) | **简体中文**

科大讯飞 Hyena Individual Identification Challenge（野生鬣狗个体识别挑战赛）
第 5 名方案，线上 Macro-F1 为 **0.77415**。本方案结合部位感知表征、异构视觉
骨干、无标签前景观察与分部位 raw-score 集成。

## 方法

- DINOv3-H+/16 部位感知主干：global、stripe 与 covariance descriptors；
- DINOv3 ConvNeXt-L 与 BioCLIP ViT-B/16 异构互补模型；
- SAM2.1 Hiera-Tiny 无标签前景观察；
- DogFaceNet 通用个体表征迁移，外部身份分类头不参与比赛训练；
- `head`、`left_body`、`right_body` 分部位 non-TTA raw-score 集成。

详细结构、训练目标和集成公式见 [方法说明](docs/METHOD.md)，完整配置见
[configs/final_pipeline.yaml](configs/final_pipeline.yaml)。

## 复现

### 环境配置

参考环境：Linux、Python 3.11、CUDA 12.8、PyTorch 2.9.1、RTX 3090 24GB。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make verify
```

### 数据准备

下载科大讯飞公开比赛包和官方引用的 Hyena ID 2022 图片归档：

- [科大讯飞赛事平台](https://challenge.xfyun.cn/)
- [LILA Hyena ID 2022](https://lila.science/datasets/hyena-id-2022/)
- [Kaggle mirror](https://www.kaggle.com/datasets/ashfaqsyed/hyena-dataset)

```bash
make prepare-data \
  PUBLIC_DATA_ZIP=/path/to/dataset\(public\).zip \
  HYENA_DATA_ZIP=/path/to/hyena-dataset.zip
```

数据将整理至 `work/competition/`。DogFaceNet 外部数据来自
[Zenodo 12578449](https://zenodo.org/records/12578449)（CC BY 4.0）；主流程使用
权重包中已经训练完成的通用表征初始化。

### 预训练权重

> [百度网盘：下载完整 `weights` 目录](https://pan.baidu.com/s/15SQ2cEHe5u3Z0TI1WKxdUQ?pwd=gv24)  
> 提取码：`gv24`

保持目录内容不变，并将其放在仓库同级：

```text
workspace/
├── hyena-identification/
└── weights/
```

```bash
make verify-weights
```

### 训练

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
export SAM2_REPO=$PWD/third_party/sam2
make masks
make train
```

完整流水线依次训练八个模型，单张 RTX 3090 约需 25 小时。

### 推理

`make train` 在每个模型训练结束后执行 non-TTA 测试推理，并自动完成分部位融合。
输出文件为：

```text
work/full_run/final_submission/submission.csv
work/full_run/final_submission/submission.zip
work/full_run/final_submission/reproduction_report.json
```

## 致谢与许可证

本项目使用 [timm](https://github.com/huggingface/pytorch-image-models)、
[DINOv3](https://github.com/facebookresearch/dinov3)、
[SAM2](https://github.com/facebookresearch/sam2)、
[BioCLIP](https://github.com/Imageomics/bioclip) 和 DogFaceNet 公开资源。
原创代码以 [MIT License](LICENSE) 发布，第三方资产遵循各自许可条款。
