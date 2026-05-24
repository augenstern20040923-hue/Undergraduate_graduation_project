import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    # 解析评估脚本的命令行参数。
    parser = argparse.ArgumentParser(description="Evaluate generated images with a CLIPScore-style metric.")
    parser.add_argument("--manifest", required=True, help="Path to generated_manifest.jsonl")
    parser.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def read_manifest(path):
    # 读取生成结果清单，每一行都是一条 JSON 记录。
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clipscore(model, processor, image, prompt, device):
    # 将图像和文本一起送入 CLIP，计算二者特征的余弦相似度并放大到 0~100 左右的区间。
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = torch.clamp((image_features * text_features).sum(dim=-1), min=0.0).item() * 100.0
    return score


def main():
    # 读取生成清单并检查是否有可评估的数据。
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError("Manifest is empty.")

    # 加载 CLIP 模型，用于衡量图像和提示词的语义一致性。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    model.eval()

    scores = []
    detailed = []
    # 逐张图片计算分数，并保留详细结果。
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        score = clipscore(model, processor, image, row["prompt"], device)
        scores.append(score)
        detailed.append({"image_path": row["image_path"], "prompt": row["prompt"], "clipscore": score})

    # 汇总平均分，并输出为 JSON 文件方便后续分析。
    result = {
        "manifest": str(manifest_path),
        "clip_model_id": args.clip_model_id,
        "image_count": len(scores),
        "mean_clipscore": sum(scores) / len(scores),
        "details": detailed,
    }
    out_path = manifest_path.parent / "clipscore_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "ok", "mean_clipscore": result["mean_clipscore"], "result_path": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件用于评估生成图片与提示词之间的匹配程度，输出一个类似 CLIPScore 的指标。
#
# 怎么使用：
# 1. 先确保你已经用 generate.py 生成了图片，并得到了 generated_manifest.jsonl。
# 2. 运行：
#    python scripts/evaluate_clipscore.py --manifest outputs/generated/generated_manifest.jsonl
# 3. 结果会保存为同目录下的 clipscore_results.json，其中包含平均分和每张图片的详细分数。
