import argparse
import csv
import json
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from utils import load_config, merge_overrides, read_jsonl, resolve_path, resolve_pretrained_model_name_or_path, save_json, set_seed


class JsonlImageTextDataset(Dataset):
    def __init__(self, dataset_dir, metadata_name, resolution, center_crop, random_flip, caption_dropout):
        # 读取图文配对的 JSONL 标注文件，并定义图像预处理流程。
        self.dataset_dir = dataset_dir
        self.rows = read_jsonl(dataset_dir / metadata_name)
        crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        flip = transforms.RandomHorizontalFlip(p=0.5) if random_flip else transforms.Lambda(lambda x: x)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                crop,
                flip,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.caption_dropout = caption_dropout

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # 读取单条样本，必要时随机丢弃文本，用于提升模型对空提示词的鲁棒性。
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["file_name"]).convert("RGB")
        text = row["text"]
        if self.caption_dropout > 0 and torch.rand(1).item() < self.caption_dropout:
            text = ""
        return {"pixel_values": self.image_transform(image), "text": text}


def parse_args():
    # 解析训练命令行参数，便于覆盖配置文件中的默认训练设置。
    parser = argparse.ArgumentParser(description="Train an SD1.5 LoRA model aligned to the paper setup.")
    parser.add_argument("--config", default="configs/paper_repro_6gb.json")
    parser.add_argument("--output_dir")
    parser.add_argument("--dataset_dir")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hf_endpoint")
    return parser.parse_args()


def select_dtype(name):
    # 将字符串精度配置转换为 PyTorch 对应的数据类型。
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return torch.float16


def collate_fn(examples, tokenizer):
    # 将一批样本拼接成训练 batch，并完成文本 tokenization。
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    texts = [ex["text"] for ex in examples]
    tokenized = tokenizer(
        texts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {"pixel_values": pixel_values, "input_ids": tokenized.input_ids}


def maybe_enable_xformers(unet, enabled):
    # 如果环境支持 xformers，就开启更省显存的注意力实现。
    if not enabled:
        return
    try:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception:
        print("xformers not available, continue without it.")


def save_lora_checkpoint(unet, output_dir, weight_name, config):
    # 导出当前 UNet 中的 LoRA 权重，并同步保存一份简化版配置。
    lora_dir = output_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=lora_dir,
        unet_lora_layers=get_peft_model_state_dict(unet),
        weight_name=weight_name,
        safe_serialization=True,
    )
    save_json(
        lora_dir / "adapter_config.json",
        {"target_modules": config["target_modules"], "rank": config["rank"], "lora_alpha": config["lora_alpha"]},
    )


def generate_validation_images(config, output_dir, device, dtype):
    # 训练完成后使用验证提示词生成样例图，便于快速目视检查训练效果。
    prompts = config.get("validation_prompts", [])
    if not prompts:
        return

    lora_dir = output_dir / "lora"
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model = resolve_pretrained_model_name_or_path(config["pretrained_model_name_or_path"])

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(lora_dir), weight_name=config["weight_name"])
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # 固定随机种子，确保不同验证轮次之间结果可对比。
    generator = torch.Generator(device=device.type).manual_seed(int(config["seed"]))
    rows = []
    for i, prompt in enumerate(prompts):
        result = pipe(
            prompt=prompt,
            negative_prompt=config.get("negative_prompt"),
            num_inference_steps=int(config["num_inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            num_images_per_prompt=int(config.get("num_validation_images", 1)),
            generator=generator,
        )
        for j, image in enumerate(result.images):
            filename = f"sample_{i:02d}_{j:02d}.png"
            image_path = sample_dir / filename
            image.save(image_path)
            rows.append({"prompt": prompt, "image_path": str(image_path.resolve())})

    # 保存验证图像与提示词的对应关系，方便后续评估。
    with (sample_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # 读取基础配置，并用命令行传入的值进行覆盖。
    args = parse_args()
    base_config = load_config(args.config)
    config = merge_overrides(
        base_config,
        {
            "output_dir": args.output_dir,
            "dataset_dir": args.dataset_dir,
            "rank": args.rank,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_steps": args.save_steps,
            "seed": args.seed,
            "hf_endpoint": args.hf_endpoint,
        },
    )

    # 支持通过镜像地址访问 Hugging Face。
    if config.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = config["hf_endpoint"]

    # 初始化随机种子、输出目录和配置记录文件。
    set_seed(int(config["seed"]))
    output_dir = resolve_path(config["output_dir"])
    dataset_dir = resolve_path(config["dataset_dir"])
    pretrained_model = resolve_pretrained_model_name_or_path(config["pretrained_model_name_or_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    # 准备训练设备和混合精度设置。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_dtype(config.get("mixed_precision", "fp16"))
    if device.type == "cuda" and config.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    # 加载 Stable Diffusion 各个核心模块。
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet", local_files_only=True)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler", local_files_only=True)

    # 冻结原始模型参数，只训练 LoRA 适配器参数。
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 向 UNet 注入 LoRA 结构。
    lora_config = LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=list(config["target_modules"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
    )
    unet.add_adapter(lora_config)

    if config.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()
    maybe_enable_xformers(unet, bool(config.get("use_xformers_if_available", True)))

    # 将模型移动到目标设备；LoRA 参数强制保留 fp32，避免 AMP 下梯度不稳定。
    vae.to(device=device, dtype=dtype)
    text_encoder.to(device=device, dtype=dtype)
    unet.to(device=device, dtype=dtype)
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    unet.train()

    # 构建训练数据集和数据加载器。
    dataset = JsonlImageTextDataset(
        dataset_dir=dataset_dir,
        metadata_name="metadata_train.jsonl",
        resolution=int(config["resolution"]),
        center_crop=bool(config["center_crop"]),
        random_flip=bool(config["random_flip"]),
        caption_dropout=float(config["caption_dropout"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=lambda examples: collate_fn(examples, tokenizer),
    )
    data_iter = iter(dataloader)

    # 仅优化可训练的 LoRA 参数。
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        weight_decay=float(config["adam_weight_decay"]),
        eps=float(config["adam_epsilon"]),
    )

    # 配置自动混合精度、梯度缩放和训练过程中的关键超参数。
    use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    autocast = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and dtype == torch.float16)
    grad_accum = int(config["gradient_accumulation_steps"])
    max_train_steps = int(config["max_train_steps"])
    log_every = int(config["log_every"])
    save_steps = int(config["save_steps"])

    log_path = output_dir / "logs" / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])

    progress_bar = tqdm(range(1, max_train_steps + 1), desc="Training")
    running_loss = 0.0

    # 主训练循环：采样噪声、构造扩散训练目标、反向传播并定期保存 LoRA。
    for global_step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            input_ids = batch["input_ids"].to(device=device)

            # 使用 VAE 将图像编码为 latent，再由文本编码器生成条件特征。
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 对 latent 添加噪声，训练 UNet 去预测噪声本身。
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast:
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean") / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item()

        # 梯度裁剪后执行参数更新，降低训练不稳定风险。
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], float(config["max_grad_norm"]))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_value = running_loss
        running_loss = 0.0
        progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_step, f"{loss_value:.6f}"])

        if global_step % log_every == 0:
            print(f"step={global_step} loss={loss_value:.6f}")

        # 按配置周期保存中间检查点，便于中断恢复或比较不同阶段效果。
        if global_step % save_steps == 0 or global_step == max_train_steps:
            ckpt_dir = output_dir / "checkpoints" / f"step_{global_step:04d}"
            save_lora_checkpoint(unet, ckpt_dir, config["weight_name"], config)

    # 训练结束后导出最终 LoRA、运行摘要和验证样图。
    save_lora_checkpoint(unet, output_dir, config["weight_name"], config)
    save_json(
        output_dir / "run_summary.json",
        {
            "status": "finished",
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
            "pretrained_model": str(pretrained_model),
            "device": str(device),
            "torch_dtype": str(dtype),
            "max_train_steps": max_train_steps,
            "rank": int(config["rank"]),
            "learning_rate": float(config["learning_rate"]),
            "train_batch_size": int(config["train_batch_size"]),
            "gradient_accumulation_steps": grad_accum,
        },
    )
    generate_validation_images(config, output_dir, device, dtype)
    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件是整个项目的核心训练脚本，用于基于图文数据集对 Stable Diffusion 1.5 的 UNet 进行 LoRA 微调。
#
# 怎么使用：
# 1. 先用 prepare_dataset.py 准备好数据集。
# 2. 再运行：
#    python scripts/train_lora.py --config configs/paper_repro_6gb.json
# 3. 如果要覆盖配置里的路径或超参数，可追加参数，例如：
#    python scripts/train_lora.py --dataset_dir data/ink_painting_45 --output_dir outputs/run1 --max_train_steps 500
# 4. 训练完成后，LoRA 权重会保存在 output_dir/lora，中间检查点保存在 output_dir/checkpoints，
#    验证样图和日志也会一起输出。
