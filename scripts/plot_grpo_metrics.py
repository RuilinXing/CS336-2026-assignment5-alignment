"""汇总多个 GRPO seed 的指标，并输出 SVG 曲线图。

示例：

    uv run python scripts/plot_grpo_metrics.py \
        --input-root outputs/grpo_standard \
        --output-dir outputs/grpo_standard_plots

脚本不依赖 matplotlib：每张 SVG 都包含跨 seed 的均值曲线和 95% 置信区间。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class MetricSpec:
    """一个图表对应的日志 split、字段名和显示标题。"""

    split: str
    key: str
    title: str


METRICS = {
    "loss": MetricSpec("train", "loss", "训练损失"),
    "train_reward": MetricSpec("train", "mean_reward", "训练总奖励"),
    "train_format_reward": MetricSpec("train", "mean_format_reward", "训练格式奖励"),
    "grad_norm": MetricSpec("train", "grad_norm", "梯度范数"),
    "token_entropy": MetricSpec("train", "mean_token_entropy", "Token 熵"),
    "val_reward": MetricSpec("validation", "val_reward", "验证总奖励"),
    "val_format_reward": MetricSpec("validation", "val_format_reward", "验证格式奖励"),
    "val_response_length": MetricSpec(
        "validation", "val_average_response_length", "验证平均回答长度"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input-root",
        type=Path,
        help="递归搜索该目录下所有 metrics.jsonl，例如 outputs/grpo_standard。",
    )
    source_group.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        help="直接指定一个或多个单独 seed 的输出目录。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    return parser.parse_args()


def find_metric_files(args: argparse.Namespace) -> list[Path]:
    """寻找每个 seed 输出目录中的 metrics.jsonl。"""
    if args.input_root is not None:
        return sorted(args.input_root.rglob("metrics.jsonl"))
    return [directory / "metrics.jsonl" for directory in args.input_dirs]


def load_metric_series(metrics_path: Path, spec: MetricSpec) -> dict[int, float]:
    """读取一个 run，并把同一 step 的多次训练更新先取平均。"""
    values_by_step: dict[int, list[float]] = defaultdict(list)
    with metrics_path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            record = json.loads(line)
            if record.get("split") == spec.split and spec.key in record:
                values_by_step[int(record["step"])].append(float(record[spec.key]))
    return {step: sum(values) / len(values) for step, values in values_by_step.items()}


def summarize_series(series_by_run: list[dict[int, float]]) -> list[dict[str, float | int]]:
    """对每个 step 计算跨 seed 的均值、样本标准差和 95% 置信区间。"""
    values_by_step: dict[int, list[float]] = defaultdict(list)
    for series in series_by_run:
        for step, value in series.items():
            values_by_step[step].append(value)

    summary = []
    for step in sorted(values_by_step):
        values = values_by_step[step]
        mean = sum(values) / len(values)
        if len(values) == 1:
            std = 0.0
        else:
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        confidence_radius = 1.96 * std / math.sqrt(len(values))
        summary.append(
            {
                "step": step,
                "mean": mean,
                "std": std,
                "count": len(values),
                "ci_lower": mean - confidence_radius,
                "ci_upper": mean + confidence_radius,
            }
        )
    return summary


def _scale(value: float, lower: float, upper: float, output_lower: float, output_upper: float) -> float:
    if upper == lower:
        return (output_lower + output_upper) / 2
    return output_lower + (value - lower) / (upper - lower) * (output_upper - output_lower)


def _points(summary: list[dict[str, float | int]], field: str, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> str:
    left, right = 80.0, 920.0
    top, bottom = 60.0, 470.0
    return " ".join(
        f"{_scale(float(item['step']), *x_bounds, left, right):.2f},"
        f"{_scale(float(item[field]), *y_bounds, bottom, top):.2f}"
        for item in summary
    )


def write_svg(summary: list[dict[str, float | int]], title: str, output_path: Path) -> None:
    """将均值和置信区间渲染成一张独立、可缩放的 SVG 图。"""
    if not summary:
        return
    x_bounds = (float(summary[0]["step"]), float(summary[-1]["step"]))
    y_values = [
        value
        for item in summary
        for value in (float(item["ci_lower"]), float(item["ci_upper"]))
    ]
    y_lower, y_upper = min(y_values), max(y_values)
    padding = max((y_upper - y_lower) * 0.08, 1e-6)
    y_bounds = (y_lower - padding, y_upper + padding)
    upper_points = _points(summary, "ci_upper", x_bounds, y_bounds)
    lower_points = _points(list(reversed(summary)), "ci_lower", x_bounds, y_bounds)
    mean_points = _points(summary, "mean", x_bounds, y_bounds)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="540" viewBox="0 0 1000 540">
  <rect width="100%" height="100%" fill="white"/>
  <text x="500" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(title)}</text>
  <line x1="80" y1="470" x2="920" y2="470" stroke="#333"/>
  <line x1="80" y1="60" x2="80" y2="470" stroke="#333"/>
  <polygon points="{upper_points} {lower_points}" fill="#4c78a8" fill-opacity="0.2"/>
  <polyline points="{mean_points}" fill="none" stroke="#4c78a8" stroke-width="3"/>
  <text x="500" y="520" text-anchor="middle" font-family="sans-serif" font-size="14">训练 rollout step</text>
  <text x="20" y="265" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 265)">{escape(title)}</text>
  <text x="80" y="490" text-anchor="middle" font-family="sans-serif" font-size="12">{x_bounds[0]:.0f}</text>
  <text x="920" y="490" text-anchor="middle" font-family="sans-serif" font-size="12">{x_bounds[1]:.0f}</text>
  <text x="70" y="470" text-anchor="end" font-family="sans-serif" font-size="12">{y_bounds[0]:.4g}</text>
  <text x="70" y="65" text-anchor="end" font-family="sans-serif" font-size="12">{y_bounds[1]:.4g}</text>
  <text x="920" y="50" text-anchor="end" font-family="sans-serif" font-size="12">均值 ± 95% CI</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metric_files = find_metric_files(args)
    if not metric_files:
        raise FileNotFoundError("没有找到 metrics.jsonl。请检查 --input-root 或 --input-dirs。")
    missing_files = [path for path in metric_files if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"以下日志文件不存在：{missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for metric_name in args.metrics:
        spec = METRICS[metric_name]
        summary = summarize_series([load_metric_series(path, spec) for path in metric_files])
        if summary:
            write_svg(summary, spec.title, args.output_dir / f"{metric_name}.svg")
            all_summaries[metric_name] = summary

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_runs": len(metric_files),
                "metric_files": [str(path) for path in metric_files],
                "metrics": all_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已处理 {len(metric_files)} 个 run，结果写入 {args.output_dir}")


if __name__ == "__main__":
    main()
