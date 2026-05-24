# 基于 LoRA 微调 Stable Diffusion 的水墨风格实验设计

本目录整理了毕业设计中可复现、适合上传到 GitHub 的实验设计材料，主题为“基于 LoRA 微调 Stable Diffusion 的个性化水墨图像生成”。内容包括实验配置、训练脚本、评价脚本、FID 结果、学习率消融结果和答辩用实验设计页面。

## 内容结构

- `ink_lora_repro/`：LoRA 训练、生成、检查与本地 Gradio 展示代码。
- `ink_lora_repro/configs/experiments_full/`：2192 张水墨图像大训练集实验配置。
- `ink_lora_repro/data/ink_painting_2192/dataset_manifest.json`：数据集规模与路径摘要，不包含原始图片。
- `contrast_quality/`：FID 与图像质量对比脚本、完整实验 FID 结果。
- `ablate/`：学习率消融实验配置与结果报告。
- `presentation/`：答辩中与训练、实验设计、损失曲线、超参数和 FID 分析相关的讲稿、SVG 页面和图片。
- `LoRA水墨答辩讲解PPT_15页_20260523.pptx`：15 页答辩讲解 PPT。
- `终稿_20260523.md`：论文终稿转换后的 Markdown 文本。

## 实验主线

实验围绕三类对比展开：

1. Rank 对比：比较不同 LoRA rank 对风格学习能力与生成质量的影响。
2. 训练步数对比：比较 600、1200、1800 步等设置下的 loss 与生成效果。
3. 学习率消融：固定 Rank=16、训练 1200 步，对比 1e-4、2e-4、5e-4。

评价方式包括训练 loss、生成样例、基础模型与 LoRA 模型对比，以及 FID 指标。当前完整实验 FID 中，`paper_r16_s600_6gb_full` 的 FID 最低，为 `356.3228`。

## 未包含内容

为了控制仓库体积并避免上传训练产物，本包没有包含：

- 2192 张训练集原图；
- LoRA 权重、checkpoint、safetensors 文件；
- Python 虚拟环境；
- `__pycache__` 缓存；
- Gradio 运行日志；
- `ppt-master` 工具仓库及其依赖。

这些内容如果需要长期保存，更适合放到云盘、Release 附件或专门的数据/模型存储位置。
