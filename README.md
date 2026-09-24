# Part-Aware Hyena Individual Identification

[English](README_EN.md) | **简体中文**

科大讯飞 Hyena Individual Identification Challenge（野生鬣狗个体识别挑战赛）
第四名方案，线上 Macro-F1 为 **0.77415**。本方案在严格按源图分组的验证体系下，
结合部位感知表征、异构视觉骨干、无标签前景观察与分部位 raw-score 集成。

仓库提供数据恢复、训练、推理、融合、无泄漏审计以及正式提交精确重建代码。

## 方案概述

- **部位感知主干**：以 DINOv3-H+/16 为主要骨干，联合 global、stripe 与
  covariance descriptors，并针对头部和左右躯干学习互补表征。
- **异构互补模型**：DINOv3 ConvNeXt-L 提供卷积归纳偏置，BioCLIP
  ViT-B/16 提供不同预训练语义空间。
- **前景观察**：固定 SAM2.1 Hiera-Tiny 生成无标签前景掩码，训练和测试采用
  完全一致的确定性流程。
- **异源表征迁移**：使用公开 DogFaceNet 犬脸身份数据进行通用个体表征预训练，
  迁移时丢弃外部身份分类头，并重新初始化比赛的 255 类分类参数。
- **分部位集成**：分别融合 `head`、`left_body`、`right_body` 的 raw scores，
  所有基础模型均为 non-TTA 单次前向。

模型结构、目标函数和集成公式见 [方法说明](docs/METHOD.md)。完整机器配置见
[configs/final_pipeline.yaml](configs/final_pipeline.yaml)。

## 仓库结构

```text
.
├── configs/        # 最终流水线配置
├── src/hyenaid/    # 模型、数据、损失与训练引擎
├── tools/          # 训练、推理、融合和审计工具
├── scripts/        # 端到端运行入口
├── benchmark/      # 固定 folds、指标与无泄漏检查
├── artifacts/      # 冻结 raw scores 与正式提交记录
└── docs/METHOD.md  # 方法说明
```

## 复现指南

### 1. 环境配置

参考环境：Linux、Python 3.11、CUDA 12.8、PyTorch 2.9.1、RTX 3090 24GB。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make verify
```

如宿主 CUDA 版本不同，请先从 PyTorch 官方源安装匹配的 `torch` 与
`torchvision`，再安装其余依赖。

### 2. 数据准备

下载科大讯飞赛事平台提供的公开比赛包，并下载官方引用的 Hyena ID 2022
原始图片归档：

- [科大讯飞赛事平台](https://challenge.xfyun.cn/)
- [LILA Hyena ID 2022](https://lila.science/datasets/hyena-id-2022/)
- [Kaggle mirror](https://www.kaggle.com/datasets/ashfaqsyed/hyena-dataset)

```bash
make prepare-data \
  PUBLIC_DATA_ZIP=/path/to/dataset\(public\).zip \
  HYENA_DATA_ZIP=/path/to/hyena-dataset.zip
```

脚本只恢复官方指定的 4,067 张训练裁剪图，并检查固定 source-group folds；
不会把归档中的额外鬣狗图片加入训练。

### 3. 权重准备

训练所需的公开初始化权重、SAM2 权重和 DogFace 表征预训练权重约 7.8 GB：

> [百度网盘：下载完整 `weights` 目录](https://pan.baidu.com/s/15SQ2cEHe5u3Z0TI1WKxdUQ?pwd=gv24)  
> 提取码：`gv24`

下载后保持目录内部结构不变，并将其放在仓库同级：

```text
workspace/
├── IFLYTEK-HyenaDetection2026-4th-Solution/
└── weights/
```

```bash
make verify-weights
```

如果权重不在默认位置，后续命令设置
`WEIGHTS_ROOT=/absolute/path/to/weights`。来源、固定 revision、许可证与 SHA-256
均记录在权重目录的 `SOURCE_MANIFEST.json` 和 `SHA256SUMS` 中。

### 4. 无需训练：精确重建正式提交

仓库保留了正式运行产生的 255 类 raw scores，可在无 GPU、无原始图片的情况下
精确重建已发布提交：

```bash
make reproduce
sha256sum work/frozen_rebuild/submission.csv
```

预期 CSV SHA-256：

```text
a37db3534c9476228799dd5acc8ee749cd21040980235f2fc2880820d632d678
```

这是对已发布预测的精确重建，不是加载模型 checkpoint 后重新前向推理。

### 5. 直接加载最终权重推理

当前公开权重包不包含训练完成后的八个比赛模型 checkpoint，因此本版本**不能**
跳过训练后直接对新图片执行最终模型前向。若要获得模型级预测，请运行下一节的
完整训练与推理流水线。公开的冻结 raw scores 仅用于逐字节验证正式提交。

### 6. 自行训练并推理

先生成固定前景掩码：

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
export SAM2_REPO=$PWD/third_party/sam2
make masks
```

随后完整训练八个模型；每个模型训练结束后立即执行固定 non-TTA 测试推理，
最后完成分部位融合：

```bash
make train
```

结果位于：

```text
work/full_run/final_submission/submission.csv
work/full_run/final_submission/submission.zip
work/full_run/final_submission/reproduction_report.json
```

在单张 RTX 3090 上，历史完整运行约需 25 小时。各模型的 epoch、学习率、
损失权重与执行顺序均固定在配置和入口脚本中。

## 数据与合规

- 官方训练集：4,067 张裁剪图，255 个身份；固定 fold manifest SHA-256 为
  `0470a4bb4630061ef06e92c50a74cb15ba457e476714294872ec426239a1ee6f`。
- DogFaceNet 外部数据来自 [Zenodo 12578449](https://zenodo.org/records/12578449)，
  许可证为 CC BY 4.0，共 8,363 张图、1,393 个犬只身份；其分类头不迁移。
- 外部犬只数据与官方训练、匿名测试图片的解码像素 SHA-256 精确交集为 0。
- 不使用额外 HyenaID 身份数据、同源动物 ReID 权重、测试标签、伪标签、
  匿名测试身份恢复或比赛 checkpoint 暖启动。
- SAM2 仅执行无标签前景分割；匿名测试图不进入任何有监督训练步骤。

本方案将 DogFace 预训练用于不同物种、不同身份空间的通用个体表征学习，并完整
披露来源、许可与迁移边界。最终规则解释以组委会为准。

## 致谢

本项目使用或参考了 [timm](https://github.com/huggingface/pytorch-image-models)、
[DINOv3](https://github.com/facebookresearch/dinov3)、
[SAM2](https://github.com/facebookresearch/sam2)、
[BioCLIP](https://github.com/Imageomics/bioclip) 与 DogFaceNet。第三方数据和权重
遵循各自发布页面的许可条款。

## 许可证

本仓库原创代码以 [MIT License](LICENSE) 发布。比赛数据、外部数据、预训练权重
及第三方代码不受该许可证覆盖，分别遵循其原始许可与比赛规则。
