import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from quality_metrics import (
    build_inception_model,
    calculate_fid,
    choose_device,
    extract_inception_features,
    list_images,
    resolve_dir,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT / "ink_lora_repro"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID for full-dataset LoRA experiments.")
    parser.add_argument(
        "--reference_dir",
        default=str(PROJECT_ROOT / "data" / "ink_painting_2192" / "images"),
        help="Real/reference image directory.",
    )
    parser.add_argument(
        "--config_dir",
        default=str(PROJECT_ROOT / "configs" / "experiments_full"),
        help="Directory containing full experiment JSON configs.",
    )
    parser.add_argument(
        "--experiments_root",
        default=str(PROJECT_ROOT / "outputs" / "experiments_full"),
        help="Directory containing full experiment outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_ROOT / "results_full_experiment_fid"),
        help="Directory for generated JSON, Markdown, and plot outputs.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--image_source",
        default="samples",
        choices=["samples"],
        help="Generated image source under each experiment output directory.",
    )
    parser.add_argument(
        "--max_reference_images",
        type=int,
        help="Optional cap for reference images, useful for quick smoke tests.",
    )
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def config_files(config_dir):
    files = sorted(Path(config_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No experiment config files found in: {config_dir}")
    return files


def resolve_output_dir(config, experiments_root):
    configured = Path(config["output_dir"])
    if configured.is_absolute():
        return configured
    return Path(experiments_root) / configured.name


def generated_dir_for_config(config, experiments_root, image_source):
    output_dir = resolve_output_dir(config, experiments_root)
    return output_dir / image_source


def maybe_limit(paths, max_count):
    if max_count is None:
        return paths
    if max_count < 2:
        raise ValueError("--max_reference_images must be at least 2 when provided.")
    return paths[:max_count]


def fid_quality_label(fid, best_fid):
    gap = (fid / best_fid - 1.0) * 100.0
    if gap < 1e-9:
        return "当前最优"
    if gap <= 5:
        return "接近最优"
    if gap <= 15:
        return "中等差距"
    return "差距较大"


def format_lr(value):
    return f"{value:.0e}" if value is not None else "-"


def make_plot(rows, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    sorted_rows = sorted(rows, key=lambda row: row["fid"])
    labels = [f"r{row['rank']}/s{row['max_train_steps']}" for row in sorted_rows]
    values = [row["fid"] for row in sorted_rows]
    colors = ["#2F7D62" if index == 0 else "#7AA6C2" for index, _ in enumerate(sorted_rows)]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Full Experiment FID Comparison")
    ax.set_xlabel("LoRA rank / train steps")
    ax.set_ylabel("FID (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    plot_path = output_dir / "full_experiment_fid_bar.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def build_report(results, output_dir, plot_path):
    rows = sorted(results["rows"], key=lambda row: row["fid"])
    best = rows[0]
    worst = rows[-1]
    best_fid = best["fid"]
    lines = [
        "# 新实验 FID 图像质量对比分析",
        "",
        "## 评价设置",
        "",
        f"- 真实参考集：`{results['reference_dir']}`",
        f"- 参考图像数：{results['reference_image_count']}",
        f"- 生成图来源：每个实验输出目录下的 `{results['image_source']}`",
        f"- FID 特征模型：torchvision Inception v3 ImageNet 权重，取最终特征层",
        f"- 运行设备：`{results['device']}`",
        f"- 生成时间：{results['created_at']}",
        "",
        "## FID 对比表",
        "",
        "| 排名 | 实验配置 | Rank | 训练步数 | 学习率 | 生成图数 | FID ↓ | 相对最优差距 | 判定 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for index, row in enumerate(rows, start=1):
        gap = (row["fid"] / best_fid - 1.0) * 100.0
        lines.append(
            "| "
            f"{index} | "
            f"`{row['experiment']}` | "
            f"{row['rank']} | "
            f"{row['max_train_steps']} | "
            f"{format_lr(row['learning_rate'])} | "
            f"{row['generated_image_count']} | "
            f"{row['fid']:.4f} | "
            f"{gap:.2f}% | "
            f"{fid_quality_label(row['fid'], best_fid)} |"
        )

    lines.extend(
        [
            "",
            "## 结果解读",
            "",
            f"1. FID 最低的是 `{best['experiment']}`，FID={best['fid']:.4f}，说明在当前生成样例集合上，它与真实水墨图像分布最接近。",
            f"2. FID 最高的是 `{worst['experiment']}`，FID={worst['fid']:.4f}，相对最优高 {(worst['fid'] / best_fid - 1.0) * 100.0:.2f}%。",
        ]
    )

    rank_1200 = [row for row in rows if row["max_train_steps"] == 1200]
    if len(rank_1200) >= 3:
        rank_best = min(rank_1200, key=lambda row: row["fid"])
        rank_worst = max(rank_1200, key=lambda row: row["fid"])
        lines.append(
            f"3. 在 1200 步设置下，Rank={rank_best['rank']} 的 FID 最低，Rank={rank_worst['rank']} 的 FID 最高，说明当前样例中并非单纯 Rank 越大越好。"
        )

    rank16 = [row for row in rows if row["rank"] == 16]
    if len(rank16) >= 3:
        step_best = min(rank16, key=lambda row: row["fid"])
        step_worst = max(rank16, key=lambda row: row["fid"])
        lines.append(
            f"4. 在 Rank=16 设置下，{step_best['max_train_steps']} 步的 FID 最低，{step_worst['max_train_steps']} 步的 FID 最高，可作为训练步数选择的客观参考。"
        )

    lines.extend(
        [
            "",
            "## 论文/汇报建议",
            "",
            "FID 越低表示生成图像集合与真实水墨数据集在 Inception 特征分布上越接近。基于本轮结果，可以把 FID 最低的配置作为当前图像质量最优设置，并结合样例图说明其水墨笔触、山水结构和整体色调更接近真实数据分布。",
            "",
            "需要注意：当前每个实验只有 3 张生成样例参与 FID，统计稳定性有限；它适合做阶段性对比和趋势判断。正式论文或最终实验建议每个配置至少生成 30 张以上，最好 50-100 张，再重新计算 FID。",
        ]
    )

    if plot_path:
        lines.extend(["", "## 图表输出", "", f"![Full Experiment FID Bar]({plot_path})"])

    report_path = output_dir / "full_experiment_fid_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    reference_dir = resolve_dir(args.reference_dir)
    config_dir = resolve_dir(args.config_dir)
    experiments_root = resolve_dir(args.experiments_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_images = maybe_limit(list_images(reference_dir), args.max_reference_images)
    device = choose_device(args.device)
    model = build_inception_model(device)
    reference_features = extract_inception_features(reference_images, model, device, args.batch_size)

    rows = []
    for config_path in config_files(config_dir):
        config = read_json(config_path)
        image_dir = generated_dir_for_config(config, experiments_root, args.image_source)
        generated_images = list_images(image_dir)
        generated_features = extract_inception_features(generated_images, model, device, args.batch_size)
        fid = calculate_fid(reference_features, generated_features)
        rows.append(
            {
                "experiment": config_path.stem,
                "config_path": str(config_path),
                "generated_dir": str(image_dir),
                "rank": int(config["rank"]),
                "max_train_steps": int(config["max_train_steps"]),
                "learning_rate": float(config["learning_rate"]),
                "generated_image_count": len(generated_images),
                "fid": fid,
                "fid_4dp": round(fid, 4),
            }
        )

    rows = sorted(rows, key=lambda row: (row["rank"], row["max_train_steps"]))
    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "reference_dir": str(reference_dir),
        "reference_image_count": len(reference_images),
        "image_source": args.image_source,
        "rows": rows,
    }

    result_path = output_dir / "full_experiment_fid_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_path = make_plot(rows, output_dir)
    report_path = build_report(results, output_dir, plot_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "result_path": str(result_path),
                "report_path": str(report_path),
                "plot_path": str(plot_path) if plot_path else None,
                "best": min(rows, key=lambda row: row["fid"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
