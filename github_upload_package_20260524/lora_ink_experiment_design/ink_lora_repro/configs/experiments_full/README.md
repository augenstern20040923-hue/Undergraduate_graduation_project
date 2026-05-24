# Full-dataset experiment configs

These configs keep the original five 6GB experiment parameter groups, switch the
dataset to `data/ink_painting_2192`, and scale the training length for the full
dataset. With batch size 1 and gradient accumulation 4, 2192 images are about
548 optimizer steps per epoch.

| Config | Rank | Steps | Output |
| --- | ---: | ---: | --- |
| `paper_r08_s1200_6gb_full.json` | 8 | 1200 | `outputs/experiments_full/paper_r08_s1200_6gb_full` |
| `paper_r16_s600_6gb_full.json` | 16 | 600 | `outputs/experiments_full/paper_r16_s600_6gb_full` |
| `paper_r16_s1200_6gb_full.json` | 16 | 1200 | `outputs/experiments_full/paper_r16_s1200_6gb_full` |
| `paper_r16_s1800_6gb_full.json` | 16 | 1800 | `outputs/experiments_full/paper_r16_s1800_6gb_full` |
| `paper_r32_s1200_6gb_full.json` | 32 | 1200 | `outputs/experiments_full/paper_r32_s1200_6gb_full` |
