import json
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | None) -> Path | None:
    # 将相对路径统一解析为相对于项目根目录的绝对路径。
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_pretrained_model_name_or_path(model_name_or_path: str) -> str:
    path = resolve_path(model_name_or_path)
    if path and path.exists():
        return str(path)

    if "/" not in model_name_or_path:
        return model_name_or_path

    cache_name = "models--" + model_name_or_path.replace("/", "--")
    snapshots_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if not snapshots_dir.exists():
        return model_name_or_path

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return model_name_or_path

    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    return str(latest)


def load_config(config_path: str | Path) -> dict:
    # 读取 JSON 配置文件。
    path = resolve_path(str(config_path))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict) -> None:
    # 保存普通 JSON 文件；如果父目录不存在则自动创建。
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    # 将多条字典记录按 JSONL 格式逐行写入。
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    # 读取 JSONL 文件并返回字典列表。
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    # 同时固定 Python、NumPy 和 PyTorch 的随机种子，提升复现性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_overrides(config: dict, overrides: dict) -> dict:
    # 用命令行传入的非空参数覆盖配置文件中的默认值。
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


# 代码作用：
# 这个文件提供整个项目共用的工具函数，包括路径解析、配置读写、JSONL 读写、随机种子设置和配置覆盖。
#
# 怎么使用：
# 1. 在其他脚本里通过 `from utils import ...` 或 `from scripts.utils import ...` 导入需要的函数。
# 2. 常见用途包括：
#    `load_config()` 读取配置，
#    `resolve_path()` 处理相对路径，
#    `set_seed()` 固定随机种子，
#    `save_json()` / `save_jsonl()` 保存结果文件。
