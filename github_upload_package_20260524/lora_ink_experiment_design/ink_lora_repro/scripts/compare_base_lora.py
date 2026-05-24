import argparse
import gc
import json
import os
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from utils import load_config, merge_overrides, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Markdown comparison table for base Stable Diffusion and LoRA results.")
    parser.add_argument("--config", default="configs/paper_repro_6gb.json")
    parser.add_argument("--lora_dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts_file")
    parser.add_argument("--output_dir", default="outputs/comparisons/base_vs_lora")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--negative_prompt")
    parser.add_argument("--hf_endpoint")
    return parser.parse_args()


def load_prompts(args, config):
    if args.prompt:
        return [args.prompt]
    if args.prompts_file:
        with resolve_path(args.prompts_file).open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return list(config.get("validation_prompts", []))


def make_pipe(config, device, dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_variant(pipe, prompts, output_dir, config, variant_name, base_seed, num_images):
    rows = []
    image_index = 0
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    for prompt_index, prompt in enumerate(prompts):
        for repeat_index in range(num_images):
            image_seed = base_seed + prompt_index * num_images + repeat_index
            generator = torch.Generator(device=pipe.device.type).manual_seed(image_seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=config.get("negative_prompt"),
                guidance_scale=float(config.get("guidance_scale", 7.5)),
                num_inference_steps=int(config.get("num_inference_steps", 30)),
                num_images_per_prompt=1,
                generator=generator,
            )
            filename = f"{variant_name}_{image_index:03d}.png"
            image_path = variant_dir / filename
            result.images[0].save(image_path)
            rows.append(
                {
                    "index": image_index,
                    "prompt": prompt,
                    "seed": image_seed,
                    "model_variant": variant_name,
                    "image_path": str(image_path.resolve()),
                }
            )
            image_index += 1

    manifest_path = variant_dir / "generated_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def markdown_image(path, markdown_dir):
    relative_path = Path(path).resolve().relative_to(markdown_dir.resolve()).as_posix()
    return f'<img src="{relative_path}" width="220">'


def escape_markdown_cell(text):
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def write_comparison_table(output_dir, config, base_rows, lora_rows):
    table_path = output_dir / "comparison_table.md"
    manifest_path = output_dir / "comparison_manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as f:
        for base_row, lora_row in zip(base_rows, lora_rows):
            row = {
                "index": base_row["index"],
                "prompt": base_row["prompt"],
                "seed": base_row["seed"],
                "base_image_path": base_row["image_path"],
                "lora_image_path": lora_row["image_path"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Stable Diffusion 基础模型与 LoRA 微调模型生成效果对比",
        "",
        f"- 基础模型：`{config['pretrained_model_name_or_path']}`",
        f"- LoRA 权重：`{config.get('lora_dir', '')}`",
        f"- 随机种子起点：`{base_rows[0]['seed'] if base_rows else ''}`",
        f"- 推理步数：`{config.get('num_inference_steps', 30)}`",
        f"- Guidance Scale：`{config.get('guidance_scale', 7.5)}`",
        "",
        "| 序号 | 提示词 | 未微调 Stable Diffusion 基础模型 | LoRA 微调模型 |",
        "| --- | --- | --- | --- |",
    ]

    for base_row, lora_row in zip(base_rows, lora_rows):
        lines.append(
            "| "
            f"{base_row['index'] + 1} | "
            f"{escape_markdown_cell(base_row['prompt'])} | "
            f"{markdown_image(base_row['image_path'], output_dir)} | "
            f"{markdown_image(lora_row['image_path'], output_dir)} |"
        )

    lines.extend(
        [
            "",
            "## 论文中可从以下角度描述",
            "",
            "- 风格贴合度：观察是否呈现水墨笔触、淡墨层次、留白和传统山水构图。",
            "- 语义一致性：观察主体是否符合提示词，例如山水、梅花、喜鹊、古琴、松树等元素。",
            "- 细节稳定性：观察是否出现模糊、结构错乱、文字水印或过饱和颜色。",
            "- 对比结论：若 LoRA 结果更接近中国水墨风格，可说明微调增强了模型对目标风格域的表达能力。",
        ]
    )

    table_path.write_text("\n".join(lines), encoding="utf-8")
    return table_path, manifest_path


def main():
    args = parse_args()
    base_config = load_config(args.config)
    config = merge_overrides(
        base_config,
        {
            "lora_dir": args.lora_dir,
            "output_dir": args.output_dir,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt,
            "hf_endpoint": args.hf_endpoint,
        },
    )

    if config.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = config["hf_endpoint"]

    prompts = load_prompts(args, config)
    if not prompts:
        raise ValueError("No prompt was provided, and config validation_prompts is empty.")

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = resolve_path(config.get("lora_dir", str(resolve_path(base_config["output_dir"]) / "lora")))
    config["lora_dir"] = str(lora_dir)
    base_seed = 42 if config.get("seed") is None else int(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = make_pipe(config, device, dtype)
    base_rows = generate_variant(pipe, prompts, output_dir, config, "base", base_seed, int(args.num_images))

    pipe.load_lora_weights(str(lora_dir), weight_name=config["weight_name"])
    lora_rows = generate_variant(pipe, prompts, output_dir, config, "lora", base_seed, int(args.num_images))
    del pipe
    cleanup_memory()

    table_path, manifest_path = write_comparison_table(output_dir, config, base_rows, lora_rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "comparison_table": str(table_path),
                "comparison_manifest": str(manifest_path),
                "image_count_per_variant": len(base_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
