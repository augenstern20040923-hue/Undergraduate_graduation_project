# 2026-05-22 大训练集学习率消融实验结果

本次实验使用 `data/ink_painting_2192` 大训练集，共 2192 张图像。三组实验固定 Rank=16、训练步数=1200、随机种子=42、batch size=1、gradient accumulation steps=4，只改变 learning rate。

| 学习率 | Rank | 训练步数 | 最终 Loss | 最低 Loss | 最低 Loss 步数 | 后 100 步平均 Loss | FID ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 16 | 1200 | 0.231827 | 0.005093 | 452 | 0.155495 | 342.1240 |
| 2e-4 | 16 | 1200 | 0.231671 | 0.005074 | 452 | 0.155390 | 395.1786 |
| 5e-4 | 16 | 1200 | 0.231410 | 0.005075 | 452 | 0.155298 | 386.7016 |

输出目录：

- 1e-4：`C:\Users\38230\Desktop\1\ink_lora_repro\outputs\ablations_full\learning_rate_20260522\lr_1e-4_r16_s1200_full`
- 2e-4：`C:\Users\38230\Desktop\1\ink_lora_repro\outputs\ablations_full\learning_rate_20260522\lr_2e-4_r16_s1200_full`
- 5e-4：`C:\Users\38230\Desktop\1\ink_lora_repro\outputs\ablations_full\learning_rate_20260522\lr_5e-4_r16_s1200_full`

图表：

- Loss 曲线：`C:\Users\38230\Desktop\1\ablate\results\learning_rate_full_20260522_loss_curve.png`
- 样例对照：`C:\Users\38230\Desktop\1\ablate\results\learning_rate_full_20260522_contact_sheet.jpg`
- 指标 JSON：`C:\Users\38230\Desktop\1\ablate\results\learning_rate_full_20260522_metrics.json`
- FID JSON：`C:\Users\38230\Desktop\1\ablate\results\learning_rate_full_20260522_fid.json`
