import argparse
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a loss curve from pasted step,loss CSV text.")
    parser.add_argument("--input", help="Optional input CSV/TXT path. If omitted, paste CSV text in the terminal.")
    parser.add_argument("--output", default="outputs/loss_curve_from_input.png", help="Output PNG path.")
    parser.add_argument("--title", default="Training Loss Curve", help="Figure title.")
    parser.add_argument("--smooth", type=int, default=0, help="Moving-average window. Example: --smooth 50")
    return parser.parse_args()


def read_pasted_csv() -> str:
    print("Paste CSV data in this format:")
    print("step,loss")
    print("1,0.300695")
    print("2,0.156811")
    print()
    print("After pasting, press Enter on an empty line to generate the figure.")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def parse_loss_csv(raw: str):
    if not raw.strip():
        raise ValueError("No CSV data was provided.")

    reader = csv.DictReader(io.StringIO(raw.strip()))
    if not reader.fieldnames or "step" not in reader.fieldnames or "loss" not in reader.fieldnames:
        raise ValueError("CSV header must contain step and loss columns.")

    steps = []
    losses = []
    for row_number, row in enumerate(reader, start=2):
        try:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
        except Exception as exc:
            raise ValueError(f"Invalid data at CSV line {row_number}: {row}") from exc

    if not steps:
        raise ValueError("No valid data rows were found.")
    return steps, losses


def moving_average(values, window: int):
    if window <= 1:
        return values
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        averaged.append(sum(segment) / len(segment))
    return averaged


def plot_loss_curve(steps, losses, output_path: Path, title: str, smooth: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.2))
    if smooth and smooth > 1:
        smoothed = moving_average(losses, smooth)
        plt.plot(steps, losses, color="#2563eb", linewidth=0.8, alpha=0.22, label="Raw loss")
        plt.plot(steps, smoothed, color="#dc2626", linewidth=2.2, label=f"Moving average ({smooth})")
        plt.legend()
    else:
        plt.plot(steps, losses, color="#2563eb", linewidth=1.8, marker="o", markersize=3)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
    else:
        raw = read_pasted_csv()
    steps, losses = parse_loss_csv(raw)
    output_path = Path(args.output)
    plot_loss_curve(steps, losses, output_path, args.title, args.smooth)
    print(f"Loss curve saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
