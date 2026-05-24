# 基于 LoRA 微调 Stable Diffusion 的个性化水墨图像生成研究

这是我的本科毕业设计项目仓库，课题围绕“基于 LoRA 微调 Stable Diffusion 的个性化图像生成研究”展开。项目以中国水墨画风格图像生成为应用场景，在 Stable Diffusion v1.5 基础模型上引入 LoRA 参数高效微调方法，使模型能够在较低显存和较少训练参数的条件下学习特定水墨风格，并通过生成样例、训练损失曲线和 FID 指标对微调效果进行分析。

## 项目简介

Stable Diffusion 具备较强的文本到图像生成能力，但通用基础模型在特定艺术风格上的表现往往不够稳定，尤其是在中国水墨画这类强调墨色层次、留白构图和笔触质感的场景中，直接使用基础模型生成的图像容易出现风格不统一、细节表达不符合目标风格等问题。

本项目采用 LoRA（Low-Rank Adaptation）对 Stable Diffusion 进行轻量化微调。LoRA 不直接更新基础模型全部参数，而是在注意力层中加入低秩可训练矩阵，使模型在保留原有生成能力的同时学习目标水墨风格。相比全量微调，这种方法显著降低了训练成本和显存需求，更适合本科毕业设计中的本地复现实验。

## 研究目标

- 构建可用于水墨风格图像生成的 LoRA 微调流程。
- 在本地环境中完成 Stable Diffusion v1.5 的训练、生成和对比实验。
- 分析 LoRA rank、训练步数和学习率等超参数对生成效果的影响。
- 使用 loss 曲线、生成样例和 FID 指标评价模型效果。
- 整理一套可复现的毕业设计实验代码、配置、结果和答辩材料。

## 技术路线

项目整体流程如下：

1. 数据准备：整理水墨风格图像数据集，并生成训练所需的 metadata 文件。
2. 模型选择：使用 `runwayml/stable-diffusion-v1-5` 作为基础模型。
3. LoRA 注入：在 U-Net 注意力层中加入低秩适配参数，冻结基础模型主体参数。
4. 模型训练：根据不同 rank、训练步数和学习率设置多组实验。
5. 图像生成：使用相同或相近提示词生成水墨风格样例。
6. 结果评价：结合训练 loss、可视化样例、基础模型对比和 FID 指标进行分析。
7. 系统展示：提供本地脚本和简单界面，用于加载 LoRA 权重并生成图像。

## 实验设计

本项目的实验分为完整数据集实验和学习率消融实验两部分。

### 完整数据集实验

完整数据集实验使用 `data/ink_painting_2192`，共 2192 张水墨图像。实验保留原 6GB 显存环境下可运行的参数组合，并将训练长度扩展到完整数据集条件下。由于 batch size 为 1，梯度累积步数为 4，因此 2192 张图像约对应 548 个 optimizer step。

| 实验配置 | Rank | 训练步数 | 学习率 |
| --- | ---: | ---: | ---: |
| `paper_r08_s1200_6gb_full` | 8 | 1200 | 2e-4 |
| `paper_r16_s600_6gb_full` | 16 | 600 | 2e-4 |
| `paper_r16_s1200_6gb_full` | 16 | 1200 | 2e-4 |
| `paper_r16_s1800_6gb_full` | 16 | 1800 | 2e-4 |
| `paper_r32_s1200_6gb_full` | 32 | 1200 | 2e-4 |

### 学习率消融实验

学习率消融实验固定 `Rank=16`、训练步数 `1200`、随机种子 `42`，对比三组学习率：

| 学习率 | Rank | 训练步数 | 最终 loss | 后 100 步平均 loss | FID↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 16 | 1200 | 0.231827 | 0.155495 | 342.1240 |
| 2e-4 | 16 | 1200 | 0.231671 | 0.155390 | 395.1786 |
| 5e-4 | 16 | 1200 | 0.231410 | 0.155298 | 386.7016 |

从 loss 看，三组学习率都能稳定完成训练，最终 loss 差距较小；从 FID 和生成样例看，`1e-4` 在本组实验中的综合表现更好。

## 实验结果

完整数据集实验的 FID 对比结果如下：

| 排名 | 实验配置 | Rank | 训练步数 | 生成图数 | FID↓ | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `paper_r16_s600_6gb_full` | 16 | 600 | 3 | 356.3228 | 当前最优 |
| 2 | `paper_r08_s1200_6gb_full` | 8 | 1200 | 3 | 365.7302 | 接近最优 |
| 3 | `paper_r16_s1800_6gb_full` | 16 | 1800 | 3 | 387.0776 | 中等差距 |
| 4 | `paper_r16_s1200_6gb_full` | 16 | 1200 | 3 | 401.7019 | 中等差距 |
| 5 | `paper_r32_s1200_6gb_full` | 32 | 1200 | 3 | 503.0517 | 差距较大 |

在当前生成样例集合上，`paper_r16_s600_6gb_full` 的 FID 最低，说明它生成的样例与真实水墨图像集的特征分布最接近。同时也可以看到，在本实验中并不是 Rank 越大、训练步数越多就一定越好，超参数需要结合生成样例和量化指标共同判断。

> 注意：当前每个完整实验配置只有 3 张生成样例参与 FID 计算，因此结果更适合用于阶段性对比和趋势分析。若用于正式论文最终结论，建议每个配置生成 30 张以上，最好 50-100 张，再重新计算 FID。

## 仓库结构

```text
.
├─ ink_lora_repro/
│  ├─ app.py
│  ├─ simple_app.py
│  ├─ requirements.txt
│  ├─ configs/experiments_full/
│  ├─ data/ink_painting_2192/dataset_manifest.json
│  └─ scripts/
├─ contrast_quality/
│  ├─ analyze_full_experiment_fid.py
│  ├─ quality_metrics.py
│  └─ results_full_experiment_fid/
├─ ablate/
│  ├─ configs_full_lr/
│  └─ results/
├─ presentation/
│  ├─ notes/
│  ├─ svg_final/
│  └─ images/
├─ LoRA水墨答辩讲解PPT_15页_20260523.pptx
└─ 终稿_20260523.md
```

## 环境依赖

主要依赖包括：

- Python 3.10+
- PyTorch
- diffusers
- transformers
- accelerate
- peft
- safetensors
- gradio
- torchvision

安装依赖：

```powershell
cd ink_lora_repro
python -m pip install -r requirements.txt
```

如果需要运行 FID 评价脚本：

```powershell
cd contrast_quality
python -m pip install -r requirements.txt
```

## 运行方式

训练完整数据集实验：

```powershell
cd ink_lora_repro
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r16_s600_6gb_full.json
```

生成样例图像：

```powershell
python .\scripts\generate.py `
  --config .\configs\experiments_full\paper_r16_s600_6gb_full.json `
  --prompt "traditional Chinese ink wash landscape, misty mountains, river, elegant brushwork" `
  --num_images 3
```

计算完整实验 FID：

```powershell
cd ..\contrast_quality
python .\analyze_full_experiment_fid.py
```

启动本地界面：

```powershell
cd ..\ink_lora_repro
python .\app.py `
  --config .\configs\experiments_full\paper_r16_s600_6gb_full.json `
  --lora_dir .\outputs\experiments_full\paper_r16_s600_6gb_full\lora
```

## 项目说明

由于 GitHub 仓库体积限制，本仓库没有包含以下内容：

- 2192 张训练集原图；
- LoRA 权重文件；
- checkpoint 中间结果；
- Python 虚拟环境；
- `__pycache__` 缓存；
- 本地运行日志。

如需完整复现实验，需要自行准备水墨图像数据集，并按照配置文件中的路径组织数据。模型权重、训练集和完整输出结果更适合通过云盘、Release 附件或专门的数据/模型存储平台保存。

## 毕业设计总结

本项目验证了 LoRA 微调在水墨风格图像生成任务中的可行性。通过较小规模的可训练参数，模型能够在 Stable Diffusion 原有生成能力基础上学习更稳定的水墨风格特征。实验结果表明，合适的 rank、训练步数和学习率设置对最终生成质量有明显影响，不能仅依据训练 loss 判断模型优劣，而应结合生成样例和 FID 等指标综合分析。
