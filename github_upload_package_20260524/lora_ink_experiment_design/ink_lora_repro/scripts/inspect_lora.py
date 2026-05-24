import argparse
import json
from pathlib import Path

from safetensors import safe_open


def parse_args():
    # 解析要检查的 LoRA 权重路径，以及可选的输出路径。
    parser = argparse.ArgumentParser(description="Inspect a LoRA safetensors file and export a readable summary.")
    parser.add_argument("--weights", required=True, help="Path to .safetensors file")
    parser.add_argument("--output", help="Optional output JSON path")
    return parser.parse_args()


def main():
    # 规范化输入输出路径。
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    output_path = Path(args.output).resolve() if args.output else weights_path.with_suffix(".summary.json")

    tensors = []
    # 逐个读取 safetensors 中的张量信息，只提取名称、形状和数据类型。
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            tensors.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )

    # 将权重摘要写出为 JSON，方便查看 LoRA 中有哪些层被保存了。
    summary = {
        "weights_path": str(weights_path),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"status": "ok", "output": str(output_path), "tensor_count": len(tensors)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# 代码作用：
# 这个文件用于查看一个 LoRA `.safetensors` 权重文件里保存了哪些张量，以及每个张量的形状和类型。
#
# 怎么使用：
# 1. 运行：
#    python scripts/inspect_lora.py --weights outputs/xxx/lora/pytorch_lora_weights.safetensors
# 2. 如果不传 --output，默认会在同目录生成一个 `.summary.json` 文件。
# 3. 这个脚本适合用于检查 LoRA 是否成功导出、导出的层数量是否符合预期。
