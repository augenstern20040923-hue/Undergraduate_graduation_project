# 论文到项目的映射

## 论文中的关键实现

- 数据主题：中国水墨画
- 数据规模：45 张
- 划分方式：40 张训练，5 张验证
- 预处理：清洗、`512x512` 归一化、文本标注
- 基座模型：`Stable Diffusion v1.5`
- 微调方式：`LoRA`
- 注入模块：`U-Net` 注意力层 Query / Key
- LoRA 配置：`r=8/16/32`，`lora_alpha=r`，`dropout=0.05`
- 训练超参数：
  - `batch_size=2`
  - `gradient_accumulation_steps=4`
  - `weight_decay=1e-4`
  - `learning_rate=1e-4 / 2e-4 / 5e-4`
  - `epochs=10`
  - `max_train_steps=200`
  - `save_steps=50`
- 评价指标：`CLIPScore + 主观评价`
- 部署：`Gradio`

## 本项目中的对应实现

- 数据主题：默认用 `zqman/Text2image-ChinesePainting`，抽样为 45 张
- 文本标注：直接使用该公开数据集自带 caption；如果你后续有自己收集的 45 张图，也可以替换数据目录继续训练
- 注入模块：配置文件里用 `target_modules=["to_q","to_k"]`
- 低显存兼容：额外提供 `paper_repro_6gb.json`

## 与论文存在的可接受差异

- 论文使用的是多来源手工筛选数据；这里为了让项目能直接复现，优先采用现成可下载的近似数据集
- `diffusers` 训练时噪声调度部分采用标准训练写法；生成阶段使用 `DDIMScheduler`
- 论文代码片段里的 `attn1_q / attn2_q` 是逻辑层面的命名，实际 `diffusers` 模块名对应为 `to_q / to_k`

## 建议的复现顺序

1. 先跑 `paper_repro_6gb.json` 验证这台机器可以稳定训练
2. 再按论文做 `rank / lr / steps` 三组消融
3. 最后用 `generate.py` 和 `evaluate_clipscore.py` 复现论文中的效果展示与客观指标
