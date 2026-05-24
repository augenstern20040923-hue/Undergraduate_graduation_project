import argparse
import os
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, DoubleVar, Entry, Frame, Label, Spinbox, StringVar, Text, Tk, messagebox

import torch
from PIL import Image, ImageTk
from diffusers import DDIMScheduler, StableDiffusionPipeline

from scripts.utils import load_config, resolve_path, resolve_pretrained_model_name_or_path


class SimpleDesktopApp:
    def __init__(self, root: Tk, config: dict, lora_dir: str | None):
        self.root = root
        self.config = config
        self.lora_dir = resolve_path(lora_dir) if lora_dir else resolve_path(config["output_dir"]) / "lora"
        self.output_dir = resolve_path("outputs/simple_ui_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipe = None
        self.current_image = None
        self.current_photo = None

        self.status_var = StringVar(value="正在加载模型，请稍候…")
        self.guidance_var = DoubleVar(value=float(config.get("guidance_scale", 7.5)))
        self.steps_var = StringVar(value=str(int(config.get("num_inference_steps", 24))))
        self.seed_var = StringVar(value=str(int(config.get("seed", 42))))

        self._build_ui()
        self._load_model_async()

    def _build_ui(self) -> None:
        self.root.title("LoRA 水墨画生成器")
        self.root.geometry("980x760")

        top = Frame(self.root)
        top.pack(fill=BOTH, expand=False, padx=12, pady=12)

        Label(top, text="提示词").pack(anchor="w")
        self.prompt_box = Text(top, height=5, wrap="word")
        self.prompt_box.pack(fill=BOTH, expand=False)
        self.prompt_box.insert(
            END,
            "traditional Chinese ink wash landscape, misty mountains, river, lone boat, elegant brushwork",
        )

        Label(top, text="负面提示词").pack(anchor="w", pady=(10, 0))
        self.negative_entry = Entry(top)
        self.negative_entry.pack(fill=BOTH, expand=False)
        self.negative_entry.insert(0, self.config.get("negative_prompt", ""))

        controls = Frame(top)
        controls.pack(fill=BOTH, expand=False, pady=(10, 0))

        Label(controls, text="步数").pack(side=LEFT)
        self.steps_spin = Spinbox(controls, from_=10, to=60, textvariable=self.steps_var, width=8)
        self.steps_spin.pack(side=LEFT, padx=(6, 16))

        Label(controls, text="Guidance").pack(side=LEFT)
        self.guidance_entry = Entry(controls, textvariable=self.guidance_var, width=8)
        self.guidance_entry.pack(side=LEFT, padx=(6, 16))

        Label(controls, text="种子").pack(side=LEFT)
        self.seed_entry = Entry(controls, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side=LEFT, padx=(6, 16))

        self.generate_button = Button(controls, text="生成图片", command=self._generate_async, state="disabled")
        self.generate_button.pack(side=LEFT)

        self.save_button = Button(controls, text="保存当前图", command=self._save_current_image, state="disabled")
        self.save_button.pack(side=LEFT, padx=(10, 0))

        Label(self.root, textvariable=self.status_var, anchor="w").pack(fill=BOTH, expand=False, padx=12)

        image_frame = Frame(self.root)
        image_frame.pack(fill=BOTH, expand=True, padx=12, pady=12)

        self.image_label = Label(image_frame, text="生成结果会显示在这里", anchor="center")
        self.image_label.pack(fill=BOTH, expand=True)

    def _load_model_async(self) -> None:
        def worker():
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    resolve_pretrained_model_name_or_path(self.config["pretrained_model_name_or_path"]),
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    local_files_only=True,
                )
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                pipe.vae.enable_slicing()
                pipe.enable_attention_slicing()
                pipe = pipe.to(self.device)
                pipe.load_lora_weights(str(self.lora_dir), weight_name=self.config["weight_name"])
                self.pipe = pipe
                self.root.after(0, lambda: self._on_model_loaded())
            except Exception as exc:
                self.root.after(0, lambda: self._on_error(f"模型加载失败：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_loaded(self) -> None:
        self.status_var.set("模型加载完成，可以开始生成。")
        self.generate_button.config(state="normal")

    def _generate_async(self) -> None:
        if self.pipe is None:
            messagebox.showinfo("提示", "模型还没加载完，请稍候。")
            return

        prompt = self.prompt_box.get("1.0", END).strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示词。")
            return

        self.generate_button.config(state="disabled")
        self.status_var.set("正在生成图片，请稍候…")

        def worker():
            try:
                seed = int(self.seed_var.get().strip())
                generator = torch.Generator(device=self.device.type).manual_seed(seed)
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_entry.get().strip() or self.config.get("negative_prompt"),
                    num_inference_steps=int(self.steps_var.get().strip()),
                    guidance_scale=float(self.guidance_var.get()),
                    generator=generator,
                )
                image = result.images[0]
                self.root.after(0, lambda: self._show_image(image))
            except Exception as exc:
                self.root.after(0, lambda: self._on_error(f"生成失败：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_image(self, image: Image.Image) -> None:
        self.current_image = image
        preview = image.copy()
        preview.thumbnail((720, 720))
        self.current_photo = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.current_photo, text="")
        self.save_button.config(state="normal")
        self.generate_button.config(state="normal")
        self.status_var.set("生成完成。")

    def _save_current_image(self) -> None:
        if self.current_image is None:
            return
        filename = datetime.now().strftime("simple_ui_%Y%m%d_%H%M%S.png")
        path = self.output_dir / filename
        self.current_image.save(path)
        self.status_var.set(f"已保存到：{path}")

    def _on_error(self, message: str) -> None:
        self.generate_button.config(state="normal" if self.pipe is not None else "disabled")
        self.status_var.set(message)
        messagebox.showerror("错误", message)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Tkinter UI for the ink painting LoRA project.")
    parser.add_argument("--config", default="configs/paper_repro_6gb.json")
    parser.add_argument("--lora_dir")
    parser.add_argument("--hf_endpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    config = load_config(args.config)
    root = Tk()
    SimpleDesktopApp(root, config, args.lora_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
