import argparse
import json
import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from utils import load_config, merge_overrides, resolve_path, resolve_pretrained_model_name_or_path


def parse_args():
    # 解析图像生成脚本的参数，支持命令行覆盖配置文件中的默认值。
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion, optionally with the trained ink painting LoRA.")
    parser.add_argument("--config", default="configs/paper_repro_6gb.json")
    parser.add_argument("--lora_dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--negative_prompt")
    parser.add_argument("--hf_endpoint")
    parser.add_argument("--disable_lora", action="store_true", help="Generate with the unfine-tuned base Stable Diffusion model.")
    return parser.parse_args()


def load_prompts(args, config):
    # 提示词来源优先级：
    # 1. --prompt 单条提示词
    # 2. --prompts_file 文本文件中的多条提示词
    # 3. 配置文件中的 validation_prompts
    if args.prompt:
        return [args.prompt]
    if args.prompts_file:
        with resolve_path(args.prompts_file).open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return list(config.get("validation_prompts", []))


def main():
    # 先读取基础配置，再用命令行参数覆盖对应字段。
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
    if args.disable_lora:
        config["disable_lora"] = True

    # 可选切换 Hugging Face 镜像地址，方便在受限网络下下载模型。
    if config.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = config["hf_endpoint"]

    # 读取本次生成要用的提示词列表。
    prompts = load_prompts(args, config)
    if not prompts:
        raise ValueError("No prompt was provided, and config validation_prompts is empty.")

    # 准备输出目录和可选 LoRA 权重目录。
    output_dir = resolve_path(config.get("output_dir", "outputs/generated"))
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = resolve_path(config.get("lora_dir", str(output_dir / "lora")))

    # 加载基础模型；默认挂载训练得到的 LoRA 权重，也可以用 --disable_lora 生成未微调基座模型结果。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pretrained_model = resolve_pretrained_model_name_or_path(config["pretrained_model_name_or_path"])
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if not config.get("disable_lora", False):
        pipe.load_lora_weights(str(lora_dir), weight_name=config["weight_name"])
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    generator = None
    # 如果配置中提供了 seed，则固定随机种子，确保可复现。
    if config.get("seed") is not None:
        generator = torch.Generator(device=device.type).manual_seed(int(config["seed"]))

    manifest_path = generated_dir / "generated_manifest.jsonl"
    rows = []
    counter = 0
    # 遍历每个提示词并生成图片，同时记录生成清单。
    for prompt in prompts:
        result = pipe(
            prompt=prompt,
            negative_prompt=config.get("negative_prompt"),
            guidance_scale=float(config.get("guidance_scale", 7.5)),
            num_inference_steps=int(config.get("num_inference_steps", 30)),
            num_images_per_prompt=int(args.num_images),
            generator=generator,
        )
        for image in result.images:
            filename = f"generated_{counter:03d}.png"
            image_path = generated_dir / filename
            image.save(image_path)
            rows.append(
                {
                    "prompt": prompt,
                    "image_path": str(image_path.resolve()),
                    "model_variant": "base" if config.get("disable_lora", False) else "lora",
                }
            )
            counter += 1

    # 将“提示词-图片路径”映射写入清单文件，便于后续评估和追踪。
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"status": "ok", "generated_dir": str(generated_dir), "image_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件用于加载基础 Stable Diffusion 模型和训练好的 LoRA 权重，根据提示词批量生成图片。
#
# 怎么使用：
# 1. 先保证你已经训练好了 LoRA，并知道配置文件和 LoRA 目录位置。
# 2. 单条提示词生成示例：
#    python scripts/generate.py --prompt "a chinese ink painting landscape"
# 3. 多条提示词文件生成示例：
#    python scripts/generate.py --prompts_file outputs/xxx/validation_prompts.txt
# 4. 生成结果会保存在 output_dir/generated 目录下，同时生成 generated_manifest.jsonl。
