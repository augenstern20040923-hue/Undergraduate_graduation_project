import argparse
import os

import gradio as gr
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from scripts.utils import load_config, resolve_path, resolve_pretrained_model_name_or_path


class InkPaintingApp:
    def __init__(self, config, lora_dir):
        # 保存配置，并根据当前环境自动选择 CPU / GPU 以及推理精度。
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # 加载 Stable Diffusion 基础模型，并关闭 safety_checker 以简化本地推理流程。
        self.pipe = StableDiffusionPipeline.from_pretrained(
            resolve_pretrained_model_name_or_path(config["pretrained_model_name_or_path"]),
            torch_dtype=self.dtype,
            safety_checker=None,
            local_files_only=True,
        )
        # 切换为 DDIM 采样器，并开启切片以降低显存占用。
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.pipe = self.pipe.to(self.device)

        # 优先使用命令行传入的 LoRA 目录，否则默认读取 output_dir/lora。
        effective_lora_dir = resolve_path(lora_dir) if lora_dir else resolve_path(config["output_dir"]) / "lora"
        self.pipe.load_lora_weights(str(effective_lora_dir), weight_name=config["weight_name"])

    def generate(self, prompt, negative_prompt, steps, guidance_scale, seed):
        # 先清理提示词，避免输入空内容时直接进入推理。
        prompt = prompt.strip()
        if not prompt:
            raise gr.Error("Prompt 不能为空。")

        generator = None
        # 当 seed >= 0 时使用固定随机种子，便于结果复现。
        if seed >= 0:
            generator = torch.Generator(device=self.device.type).manual_seed(int(seed))

        # 调用扩散模型完成图像生成，并返回首张图像和运行参数说明。
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or self.config.get("negative_prompt"),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        return result.images[0], f"seed={seed}, steps={steps}, guidance_scale={guidance_scale}"


def parse_args():
    # 解析命令行参数，用于控制配置文件、LoRA 权重目录和 Web 服务端口。
    parser = argparse.ArgumentParser(description="Launch the Gradio UI for the ink painting LoRA.")
    parser.add_argument("--config", default="configs/paper_repro_6gb.json")
    parser.add_argument("--lora_dir")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--hf_endpoint")
    return parser.parse_args()


def main():
    # 读取配置，并在需要时切换 Hugging Face 下载源。
    args = parse_args()
    config = load_config(args.config)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # 初始化应用对象，并从配置中读取示例提示词供界面展示。
    app = InkPaintingApp(config, args.lora_dir)
    examples = [[prompt] for prompt in config.get("validation_prompts", [])]

    # 构建 Gradio 界面：左侧输入参数，右侧展示生成结果。
    with gr.Blocks(title="基于 LoRA 的水墨画个性化生成系统") as demo:
        gr.Markdown("## 基于 LoRA 的水墨画个性化生成系统")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", lines=4, placeholder="输入中英文提示词，通常英文提示词更稳定")
                negative_prompt = gr.Textbox(label="负面提示词", lines=2, value=config.get("negative_prompt", ""))
                steps = gr.Slider(label="采样步数", minimum=10, maximum=60, value=config.get("num_inference_steps", 24), step=1)
                guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=12.0, value=config.get("guidance_scale", 7.5), step=0.5)
                seed = gr.Number(label="随机种子", value=int(config.get("seed", 42)), precision=0)
                run_button = gr.Button("生成")
            with gr.Column():
                image = gr.Image(label="生成结果")
                info = gr.Textbox(label="运行信息")

        if examples:
            gr.Examples(examples=examples, inputs=prompt, label="示例提示词")

        run_button.click(
            fn=app.generate,
            inputs=[prompt, negative_prompt, steps, guidance, seed],
            outputs=[image, info],
        )

    # 启动本地 Web 服务。
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件用于启动一个基于 Gradio 的本地网页界面，加载训练好的水墨画 LoRA，
# 然后让你在网页里输入提示词并生成图像。
#
# 怎么使用：
# 1. 先准备好配置文件和训练输出的 LoRA 权重目录。
# 2. 在项目根目录运行：python app.py
# 3. 如需指定配置或端口，可运行：
#    python app.py --config configs/paper_repro_6gb.json --lora_dir outputs/xxx/lora --server_port 7860
# 4. 打开终端里显示的本地地址，在网页中输入 prompt 后点击按钮即可生成图片。
