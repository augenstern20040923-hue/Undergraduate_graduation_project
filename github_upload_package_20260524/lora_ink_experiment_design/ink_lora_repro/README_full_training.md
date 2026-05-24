# 全量数据集单独实验训练说明

本文件用于重新训练全量水墨画数据集版本。全量数据已经放在：

```text
C:\Users\38230\Desktop\1\ink_lora_repro\data\ink_painting_2192
```

当前全量数据共有 2192 张图片，训练脚本会读取：

```text
data\ink_painting_2192\metadata_train.jsonl
```

## 训练前先进入项目目录

每次训练前，先在 PowerShell 中执行：

```powershell
cd C:\Users\38230\Desktop\1\ink_lora_repro
```

下面每条命令都是单独训练一个实验。你不想一键训练时，就只运行其中一条。

## 实验 1：Rank 8，1200 步

用途：低 Rank 对比实验，参数量更小，训练和推理相对轻一些。

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r08_s1200_6gb_full.json --hf_endpoint https://hf-mirror.com
```

输出目录：

```text
outputs\experiments_full\paper_r08_s1200_6gb_full
```

损失日志：

```text
outputs\experiments_full\paper_r08_s1200_6gb_full\logs\train_log.csv
```

## 实验 2：Rank 16，600 步

用途：快速试跑实验，约等于完整看完数据集 1 遍，适合先确认训练流程、显存和结果是否正常。

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r16_s600_6gb_full.json --hf_endpoint https://hf-mirror.com
```

输出目录：

```text
outputs\experiments_full\paper_r16_s600_6gb_full
```

损失日志：

```text
outputs\experiments_full\paper_r16_s600_6gb_full\logs\train_log.csv
```

## 实验 3：Rank 16，1200 步

用途：推荐主实验。Rank 16 是原项目里的主参数，1200 步约等于训练 2 个 epoch，通常比 600 步更稳定。

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r16_s1200_6gb_full.json --hf_endpoint https://hf-mirror.com
```

输出目录：

```text
outputs\experiments_full\paper_r16_s1200_6gb_full
```

损失日志：

```text
outputs\experiments_full\paper_r16_s1200_6gb_full\logs\train_log.csv
```

## 实验 4：Rank 16，1800 步

用途：更充分训练实验，约等于训练 3 个 epoch。适合和 600、1200 步对比，看继续训练是否改善画风或开始过拟合。

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r16_s1800_6gb_full.json --hf_endpoint https://hf-mirror.com
```

输出目录：

```text
outputs\experiments_full\paper_r16_s1800_6gb_full
```

损失日志：

```text
outputs\experiments_full\paper_r16_s1800_6gb_full\logs\train_log.csv
```

## 实验 5：Rank 32，1200 步

用途：高 Rank 对比实验，模型表达能力更强，但更容易过拟合，也更占显存和训练时间。

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r32_s1200_6gb_full.json --hf_endpoint https://hf-mirror.com
```

输出目录：

```text
outputs\experiments_full\paper_r32_s1200_6gb_full
```

损失日志：

```text
outputs\experiments_full\paper_r32_s1200_6gb_full\logs\train_log.csv
```

## 训练顺序建议

建议先训练：

```powershell
python .\scripts\train_lora.py --config .\configs\experiments_full\paper_r16_s1200_6gb_full.json --hf_endpoint https://hf-mirror.com
```

这是最适合作为最终模型的主实验。如果时间够，再跑：

```text
Rank 16, 600 步
Rank 16, 1800 步
Rank 8, 1200 步
Rank 32, 1200 步
```

这样论文里可以分别比较训练步数和 Rank 对生成质量的影响。

## 单独打开训练好的模型桌面前端

下面的命令用于启动 `simple_app.py` 桌面版前端。运行后会直接弹出一个本地窗口，不需要打开浏览器地址。

注意：每次只打开一个模型前端。要换另一个模型，先关闭当前窗口，再运行新的命令。不要同时打开多个模型，否则这台机器很容易显存不足。

### 打开 Rank 8，1200 步模型

```powershell
python .\simple_app.py --config .\configs\experiments_full\paper_r08_s1200_6gb_full.json --lora_dir .\outputs\experiments_full\paper_r08_s1200_6gb_full\lora --hf_endpoint https://hf-mirror.com
```

### 打开 Rank 16，600 步模型

```powershell
python .\simple_app.py --config .\configs\experiments_full\paper_r16_s600_6gb_full.json --lora_dir .\outputs\experiments_full\paper_r16_s600_6gb_full\lora --hf_endpoint https://hf-mirror.com
```

### 打开 Rank 16，1200 步模型

```powershell
python .\simple_app.py --config .\configs\experiments_full\paper_r16_s1200_6gb_full.json --lora_dir .\outputs\experiments_full\paper_r16_s1200_6gb_full\lora --hf_endpoint https://hf-mirror.com
```

这是推荐优先查看的主实验模型。

### 打开 Rank 16，1800 步模型

```powershell
python .\simple_app.py --config .\configs\experiments_full\paper_r16_s1800_6gb_full.json --lora_dir .\outputs\experiments_full\paper_r16_s1800_6gb_full\lora --hf_endpoint https://hf-mirror.com
```

### 打开 Rank 32，1200 步模型

```powershell
python .\simple_app.py --config .\configs\experiments_full\paper_r32_s1200_6gb_full.json --lora_dir .\outputs\experiments_full\paper_r32_s1200_6gb_full\lora --hf_endpoint https://hf-mirror.com
```

如果窗口长时间显示正在加载模型，先等一会儿。第一次启动会加载 Stable Diffusion 基础模型和 LoRA 权重，速度取决于显卡和硬盘。

## 损失曲线怎么画

每个实验都会生成一个 CSV 文件：

```text
outputs\experiments_full\实验名\logs\train_log.csv
```

文件格式是：

```csv
step,loss
1, ...
2, ...
```

绘制损失曲线时，横轴用 `step`，纵轴用 `loss`。如果要比较五组实验，可以把五个 `train_log.csv` 画在同一张图里。

如果你想直接粘贴一段 `step,loss` 数据生成折线图，可以运行：

```powershell
python .\scripts\plot_loss_curve_from_input.py --output .\outputs\loss_curve_from_input.png
```

运行后，把类似下面的数据直接粘贴进去：

```csv
step,loss
1,0.300695
2,0.156811
3,0.083393
4,0.108262
```

粘贴完成后，按一次空行回车，脚本会生成：

```text
outputs\loss_curve_from_input.png
```

如果你想给图片换名字，可以改 `--output` 后面的路径，例如：

```powershell
python .\scripts\plot_loss_curve_from_input.py --output .\outputs\loss_rank16_1200.png --title "Rank 16 1200 Steps Loss"
```

## 训练结果在哪里

每个实验训练完成后，LoRA 权重在：

```text
outputs\experiments_full\实验名\lora
```

中间检查点在：

```text
outputs\experiments_full\实验名\checkpoints
```

验证生成图片在：

```text
outputs\experiments_full\实验名\samples
```

## 注意事项

1. 不要同时运行多个训练命令。这台机器显存有限，同时训练容易爆显存。
2. 如果只想先确认能跑通，先跑 Rank 16，600 步。
3. 如果要作为论文主结果，优先使用 Rank 16，1200 步。
4. 如果训练中断，已经写入的 `train_log.csv` 仍然可以用于查看前面步骤的损失变化。
5. 如果 Hugging Face 直连不稳定，命令里的 `--hf_endpoint https://hf-mirror.com` 不要删。
