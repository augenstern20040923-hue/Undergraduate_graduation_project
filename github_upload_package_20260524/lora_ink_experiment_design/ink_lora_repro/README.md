# 基于论文的 LoRA + Stable Diffusion 水墨画复现项目

这个项目按照你的论文《基于LoRA微调Stable Diffusion的个性化图像生成研究》落地成了一个可运行的本地复现工程，核心路线对齐为：

- 基座模型：`runwayml/stable-diffusion-v1-5`
- 微调方式：`LoRA`
- 注入位置：`U-Net` 注意力层的 `Query/Key` 投影
- 图像分辨率：`512x512`
- 最优超参数：`Rank=16`、`learning_rate=2e-4`、`max_train_steps=200`
- 训练目标：生成中国水墨 / 山水风格图像
- 界面：`Gradio`

项目默认使用 Hugging Face 数据集 `zqman/Text2image-ChinesePainting`，并抽样成和论文接近的 `45` 张图像，其中 `40` 张训练、`5` 张验证。

## 参考来源

- GitHub 基础实现思路：Hugging Face `diffusers` 官方 LoRA 文本到图像训练范式
- 数据集：`zqman/Text2image-ChinesePainting`

## 目录结构

```text
ink_lora_repro/
├─ app.py
├─ requirements.txt
├─ configs/
│  ├─ paper_repro.json
│  └─ paper_repro_6gb.json
├─ scripts/
│  ├─ utils.py
│  ├─ prepare_dataset.py
│  ├─ train_lora.py
│  ├─ generate.py
│  └─ evaluate_clipscore.py
└─ paper_alignment.md
```

## 安装依赖

在当前目录打开 PowerShell 后执行：

```powershell
cd C:\Users\38230\Desktop\1\ink_lora_repro
python -m pip install -r requirements.txt
```

## 1. 准备数据集

默认从 Hugging Face 下载一个和论文风格接近的中国山水/水墨图文数据集，并抽样 45 张：

```powershell
python .\scripts\prepare_dataset.py `
  --dataset_id zqman/Text2image-ChinesePainting `
  --output_dir .\data\ink_painting_45 `
  --limit 45 `
  --val_size 5 `
  --seed 42
```

如果你当前网络直连 Hugging Face 不稳定，可以改成：

```powershell
python .\scripts\prepare_dataset.py `
  --dataset_id zqman/Text2image-ChinesePainting `
  --output_dir .\data\ink_painting_45 `
  --limit 45 `
  --val_size 5 `
  --seed 42 `
  --hf_endpoint https://hf-mirror.com
```

执行后会得到：

- `data/ink_painting_45/images/`
- `data/ink_painting_45/metadata_train.jsonl`
- `data/ink_painting_45/metadata_val.jsonl`
- `data/ink_painting_45/dataset_manifest.json`

## 2. 按论文参数训练

如果你还没有本地缓存 `Stable Diffusion v1.5`，第一次训练前建议先确保 Hugging Face 访问正常；部分环境需要先接受模型许可并登录 `HF_TOKEN`。

标准论文配置：

```powershell
python .\scripts\train_lora.py --config .\configs\paper_repro.json
```

如果你直接在这台 `RTX 3060 Laptop 6GB` 机器上跑，建议先用低显存配置：

```powershell
python .\scripts\train_lora.py --config .\configs\paper_repro_6gb.json
```

如果 Hugging Face 直连不稳定：

```powershell
python .\scripts\train_lora.py `
  --config .\configs\paper_repro_6gb.json `
  --hf_endpoint https://hf-mirror.com
```

训练结果会保存在：

- `outputs/paper_repro/lora/`
- `outputs/paper_repro/samples/`
- `outputs/paper_repro/logs/train_log.csv`

## 3. 生成图像

```powershell
python .\scripts\generate.py `
  --config .\configs\paper_repro_6gb.json `
  --prompt "traditional Chinese ink wash landscape, misty mountains, river, lone boat, elegant brushwork" `
  --num_images 2 `
  --hf_endpoint https://hf-mirror.com
```

如果不传 `--prompt`，脚本会使用配置中的验证提示词。

## 4. 计算 CLIPScore

先用 `generate.py` 生成一批图像，它会写出 `generated_manifest.jsonl`。然后执行：

```powershell
python .\scripts\evaluate_clipscore.py `
  --manifest .\outputs\paper_repro_6gb\generated\generated_manifest.jsonl
```

## 5. 与未微调 Stable Diffusion 基础模型对比

生成论文中的“基础模型 vs LoRA 微调模型”效果对比表：

```powershell
python .\scripts\compare_base_lora.py `
  --config .\configs\experiments\paper_r16_s200_6gb.json `
  --lora_dir .\outputs\experiments\paper_r16_s200_6gb\lora `
  --output_dir .\outputs\comparisons\base_vs_paper_r16_s200_6gb `
  --num_images 1 `
  --hf_endpoint https://hf-mirror.com
```

结果会保存为：

- `outputs/comparisons/base_vs_paper_r16_s200_6gb/comparison_table.md`

如果只想生成未微调基础模型图片，可以给 `generate.py` 增加 `--disable_lora`：

```powershell
python .\scripts\generate.py `
  --config .\configs\experiments\paper_r16_s200_6gb.json `
  --output_dir .\outputs\comparisons\base_only_r16_s200 `
  --disable_lora `
  --hf_endpoint https://hf-mirror.com
```

## 6. 启动更简单的桌面界面

这是现在更推荐的方式，不走本地网页，直接打开一个 Windows 桌面窗口：

```powershell
python .\simple_app.py `
  --config .\configs\paper_repro_6gb.json `
  --lora_dir .\outputs\demo_smoke\lora `
  --hf_endpoint https://hf-mirror.com
```

## 7. 启动 Gradio 界面

如果你之后还想保留网页版，也可以继续用：

```powershell
python .\app.py `
  --config .\configs\paper_repro_6gb.json `
  --lora_dir .\outputs\paper_repro_6gb\lora `
  --hf_endpoint https://hf-mirror.com
```

## 和论文的对齐说明

- 论文写的是 `Query/Key` 注入；在 `diffusers + peft` 里对应为注意力层中的 `to_q`、`to_k`
- 论文数据集来源是多站点人工收集；这里默认给了一个现成可下载、题材高度接近的公开数据集，方便复现
- 论文实验硬件按 `12G` 显存描述；你当前机器实测是 `6GB`，因此额外提供了低显存配置
- 论文提到 LoRA 权重大约 `48MB`；本项目最终权重体积会受 `rank`、保存格式和 diffusers/peft 版本影响，量级会接近但不保证完全一致

## 论文消融实验如何复现

你论文里的三组对比实验可以直接这样跑：

1. `Rank` 对比：把 `rank` 改成 `8 / 16 / 32`
2. `学习率` 对比：把 `learning_rate` 改成 `1e-4 / 2e-4 / 5e-4`
3. `训练步数` 对比：把 `max_train_steps` 改成 `100 / 200 / 300`

最简单的做法是复制 `configs/paper_repro.json` 出三个版本分别训练。

如果你想直接用已经分好的论文实验配置，可以看：

- [configs/experiments/README.md](/C:/Users/38230/Desktop/1/ink_lora_repro/configs/experiments/README.md)
