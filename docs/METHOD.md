# 方法说明

## 摘要

本方案面向头部、左躯干和右躯干三种观察条件下的鬣狗个体分类。核心思路不是
继续扩大单一模型，而是从部位感知表征、前景观察、视觉骨干和预训练域四个方向
构建误差结构不同的分类器，再按部位进行固定 raw-score 组合。全部基础模型均从
公开通用权重或已披露的异源犬只表征权重开始完整训练，并采用 non-TTA 单次前向。

## 1. 问题定义与验证协议

给定裁剪图像 \(x\)、部位 \(p\in\{h,l,r\}\) 和 255 个训练身份，模型输出
类别分数 \(s(x,p)\in\mathbb{R}^{255}\)。评价指标分别计算三个部位的
Macro-F1，再等权平均。

训练集包含 4,067 张官方裁剪图。五折划分以原始 `source_group` 为最小单元，
同一源照片派生的裁剪不会跨越训练与验证集合。架构选择只使用固定验证折，匿名
测试图仅用于无标签前景掩码生成和最终推理。

## 2. 部位感知表征

主要分类器采用通用预训练的 DINOv3-H+/16。最后一层 patch tokens 被恢复为
二维特征图，并构建七个描述子：一个 global descriptor、两个 2-stripe
descriptors、三个 3-stripe descriptors 和一个 covariance descriptor。
这些描述子共享身份原型，同时学习部位相关的分类增量。

前 24 个 Transformer blocks 的原始参数冻结，但其中插入的 AdaptFormer 可训练；
最后 8 个 blocks 正常微调。统一基础配置如下：

| 配置 | 数值 |
|---|---:|
| 输入分辨率 | 448 × 448 |
| 采样器 | 16 identities × 4 images |
| 嵌入维度 | 512 |
| Backbone LR | 2.4e-5 |
| Head LR | 3.0e-4 |
| Layer decay | 0.82 |
| Weight decay | 0.05 |
| ArcFace scale / margin | 30.0 / 0.20 |
| Shared CE / SupCon / Triplet | 0.50 / 0.15 / 0.30 |
| Label smoothing | 0.05 |
| 随机种子 | 20260719 |

在相同主干上分别训练以下可解释变体：

| 模型名称 | 主要差异 | Epochs |
|---|---|---:|
| DINOv3-H+ 头部观察分类器 | 强化头部确定性观察 | 28 |
| DINOv3-H+ 拓扑对比分类器 | topology-aware SupCon | 37 |
| DINOv3-H+ 前景中心分类器 | centered foreground token | 36 |
| DINOv3-H+ 来源感知队列分类器 | 2,048 容量 instance queue | 32 |
| DINOv3-H+ DogFace 迁移分类器 | 异源个体表征初始化 | 33 |
| DINOv3-H+ 质量视图分类器 | deterministic quality view | 20 |

## 3. 异构视觉模型

同构 Transformer 的错误相关性较高，因此增加两类不同表征：

- **DINOv3 ConvNeXt-L 双层分类器**：融合中层与末层 feature maps，并构建
  MGN/covariance descriptors，以卷积归纳偏置补充头部识别，训练 24 epochs。
- **BioCLIP ViT-B/16 躯干分类器**：仅使用 visual tower，不加载文本塔或
  外部输出头；其生物图像预训练空间用于提供躯干方向的互补残差，训练 14 epochs。

## 4. 无标签前景观察

固定 SAM2.1 Hiera-Tiny 为官方训练图和匿名测试图离线生成二值前景掩码。
SAM2 不读取身份标签，训练和测试采用相同参数：

- source commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- checkpoint SHA-256:
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- 前景面积范围：`[0.10, 0.90]`
- 置信度阈值：`0.85`

掩码只定义确定性前景观察，不产生或修正身份标签。

## 5. DogFace 通用个体表征迁移

DogFace 表征预训练使用 8,363 张犬脸和 1,393 个犬只身份训练 16 epochs。
迁移到比赛分类器时，仅加载 backbone、adapter 和 embedding 等 638 个表征
tensors；外部身份相关的 `shared_class_weight`、`part_class_delta` 与
`part_class_available` 全部丢弃。比赛的 255 类分类参数重新初始化。

该数据来自 Zenodo 12578449（CC BY 4.0），与官方训练及匿名测试图的解码像素
SHA-256 精确交集为 0。预训练过程不读取任何比赛图像或鬣狗身份标签。

## 6. 分部位 raw-score 集成

令以下名称分别表示对应分类器输出：

- \(S_{queue}\)：DINOv3-H+ 来源感知队列分类器；
- \(S_{head}\)：DINOv3-H+ 头部观察分类器；
- \(S_{dog}\)：DINOv3-H+ DogFace 迁移分类器；
- \(S_{conv}\)：DINOv3 ConvNeXt-L 双层分类器；
- \(S_{topo}\)：DINOv3-H+ 拓扑对比分类器；
- \(S_{fg}\)：DINOv3-H+ 前景中心分类器。

基础组合为：

\[
\begin{aligned}
A_h &= 0.50S_{queue}+0.20S_{head}+0.10S_{dog}+0.20S_{conv},\\
A_l &= 0.25S_{queue}+0.5625S_{topo}+0.1875S_{fg},\\
A_r &= 0.75S_{queue}+0.25S_{topo}.
\end{aligned}
\]

令 \(S_{quality}\) 为 DINOv3-H+ 质量视图分类器，\(S_{bio}\) 为 BioCLIP
躯干分类器。最终固定残差外推为：

\[
\begin{aligned}
S_h &= 1.30A_h-0.30S_{quality},\\
S_l &= 1.15A_l-0.15S_{bio},\\
S_r &= 1.15A_r-0.15S_{bio}.
\end{aligned}
\]

所有分数的样本顺序与 255 类标签顺序在融合前强制校验，最终逐行 `argmax`。
流水线不包含 gallery 检索、测试标签校准、伪标签或匿名测试身份恢复。

## 7. 实现对应关系

- `src/hyenaid/model.py`：部位感知模型与描述子；
- `src/hyenaid/data.py`：数据增强、身份采样和前景观察；
- `src/hyenaid/losses.py`：ArcFace、SupCon 与 triplet objectives；
- `tools/train_anchor_component.py`：六个基础分类器的训练与推理；
- `tools/train_complementary_component.py`：两个残差分类器的训练与推理；
- `tools/fuse_anchor.py`：基础分部位组合；
- `tools/build_final_submission.py`：残差外推与提交生成。
