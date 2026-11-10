import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_LOGS = [Path("outputs_tests/dfine_cache_pretrained_exp/log.txt")]
METRIC_INDEXES = {
    "ap": 0,
    "ap50": 1,
}
METRIC_LABELS = {
    "ap": "AP",
    "ap50": "AP50",
}


@dataclass
class EvalRun:
    label: str
    epochs: list[int]
    metrics: dict[str, list[float]]


def resolve_log_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "log.txt"
    return path


def infer_label(log_path):
    return log_path.parent.name if log_path.name == "log.txt" else log_path.stem


def load_run(path, label=None):
    log_path = resolve_log_path(path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    epochs = []
    metrics = {name: [] for name in METRIC_INDEXES}

    with log_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{line_number} is not valid JSON.") from exc

            bbox_metrics = record.get("test_coco_eval_bbox")
            if bbox_metrics is None:
                continue

            if not isinstance(bbox_metrics, list) or len(bbox_metrics) < 2:
                raise ValueError(
                    f"{log_path}:{line_number} test_coco_eval_bbox must contain at least 2 values."
                )

            epoch = record.get("epoch", len(epochs))
            epochs.append(int(epoch))
            for metric_name, metric_index in METRIC_INDEXES.items():
                metrics[metric_name].append(float(bbox_metrics[metric_index]))

    if not epochs:
        raise ValueError(f"No test_coco_eval_bbox records found in {log_path}")

    return EvalRun(
        label=label or infer_label(log_path),
        epochs=epochs,
        metrics=metrics,
    )


def setup_matplotlib_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_runs(runs, output_path, metrics=("ap", "ap50"), title=None):
    if not runs:
        raise ValueError("At least one run is required.")

    for metric_name in metrics:
        if metric_name not in METRIC_INDEXES:
            raise ValueError(f"Unsupported metric: {metric_name}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_matplotlib_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.8 * len(metrics), 5.8))
    if len(metrics) == 1:
        axes = [axes]

    if title and hasattr(fig, "suptitle"):
        fig.suptitle(title)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric_name in zip(axes, metrics):
        for run_index, run in enumerate(runs):
            color = colors[run_index % len(colors)]
            ax.plot(
                run.epochs,
                run.metrics[metric_name],
                linestyle="-",
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=color,
                label=run.label,
            )

        ax.set_title(METRIC_LABELS[metric_name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric_name])
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_labels(labels, log_count):
    if labels is None:
        return [None] * log_count
    if len(labels) != log_count:
        raise ValueError("--labels count must match the number of logs.")
    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot AP and AP50 curves from DEIM JSONL log.txt files."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=DEFAULT_LOGS,
        help="One or more log.txt files or experiment directories. Defaults to outputs_tests/dfine_cache_pretrained_exp/log.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs_tests/coco_eval_bbox_trend.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional display labels, one per input log.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_INDEXES),
        default=["ap", "ap50"],
        help="Metrics to plot from test_coco_eval_bbox.",
    )
    parser.add_argument(
        "--title",
        help="Optional chart title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels = parse_labels(args.labels, len(args.logs))
    runs = [load_run(log, label=label) for log, label in zip(args.logs, labels)]
    plot_runs(runs, args.output, metrics=args.metrics, title=args.title)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
