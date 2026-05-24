import argparse
import json
import os
import random

from datasets import load_dataset
from PIL import Image

from utils import PROJECT_ROOT, save_json, save_jsonl


def parse_args() -> argparse.Namespace:
    # 解析数据集准备阶段需要的参数，例如数据源、采样数量、验证集大小和缩放尺寸。
    parser = argparse.ArgumentParser(description="Download and sample a Chinese painting dataset for LoRA training.")
    parser.add_argument("--dataset_id", default="zqman/Text2image-ChinesePainting")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data/ink_painting_45")
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--caption_field", default="text")
    parser.add_argument("--limit", type=int, default=45)
    parser.add_argument("--val_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--hf_endpoint", default=None, help="Optional Hugging Face mirror endpoint, e.g. https://hf-mirror.com")
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    # 将换行和多余空格清理掉，避免提示词格式不统一。
    return " ".join(text.replace("\n", " ").split())


def main() -> None:
    # 创建输出目录，并在需要时切换 Hugging Face 下载源。
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    try:
        dataset = load_dataset(args.dataset_id, split=args.split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the dataset from Hugging Face. "
            "If you are in a restricted network environment, try adding "
            "--hf_endpoint https://hf-mirror.com"
        ) from exc
    if len(dataset) < args.limit:
        raise ValueError(f"Dataset only has {len(dataset)} rows, smaller than requested limit {args.limit}.")

    # 使用固定随机种子打乱样本索引，保证采样过程可复现。
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    sampled_indices = indices[: args.limit]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    all_rows: list[dict] = []

    # 遍历采样出的数据，保存图片文件，并生成训练/验证所需的 metadata。
    for i, idx in enumerate(sampled_indices):
        sample = dataset[int(idx)]
        image = sample[args.image_field]
        caption = normalize_caption(str(sample[args.caption_field]))
        image = image.convert("RGB")
        if args.resize:
            image = image.resize((args.resize, args.resize), Image.LANCZOS)

        filename = f"{i:03d}.png"
        relative_path = f"images/{filename}"
        image.save(images_dir / filename)

        row = {
            "id": i,
            "source_index": int(idx),
            "file_name": relative_path,
            "text": caption,
        }
        all_rows.append(row)
        if i < args.val_size:
            val_rows.append(row)
        else:
            train_rows.append(row)

    # 输出训练集、验证集和全集的 JSONL 清单，以及整体数据集说明文件。
    save_jsonl(output_dir / "metadata_train.jsonl", train_rows)
    save_jsonl(output_dir / "metadata_val.jsonl", val_rows)
    save_jsonl(output_dir / "metadata_all.jsonl", all_rows)
    save_json(
        output_dir / "dataset_manifest.json",
        {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "output_dir": str(output_dir),
            "limit": args.limit,
            "val_size": args.val_size,
            "train_size": len(train_rows),
            "image_size": args.resize,
            "seed": args.seed,
            "project_root": str(PROJECT_ROOT),
        },
    )

    # 额外导出验证集提示词文本，后续可直接用于验证图片生成。
    with (output_dir / "validation_prompts.txt").open("w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(row["text"] + "\n")

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "train_size": len(train_rows), "val_size": len(val_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件用于从 Hugging Face 下载中国画数据集，随机抽样后整理成 LoRA 训练可直接使用的数据格式。
#
# 怎么使用：
# 1. 运行：
#    python scripts/prepare_dataset.py --output_dir data/ink_painting_45
# 2. 如需使用镜像，可加：
#    python scripts/prepare_dataset.py --hf_endpoint https://hf-mirror.com
# 3. 执行后会生成 images、metadata_train.jsonl、metadata_val.jsonl、metadata_all.jsonl
#    和 validation_prompts.txt 等文件，供后续训练和验证使用。
