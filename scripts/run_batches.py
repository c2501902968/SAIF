#!/usr/bin/env python
"""Unified Python entry point for experiment tasks.

Use ``python scripts/run_batches.py list`` to see the available tasks. Thin
Python entrypoints in ``scripts/`` call into this shared runner.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRY_RUN = False

TABLE2_SETTINGS = [
    "1+5@1",
    "1+10@1",
    "1+10@3",
    "1+100@3",
    "3+100@10",
    "3+1000@10",
    "10+1000@100",
    "10+10000@100",
]
MAIN_SETTINGS = ["1+100@3", "3+1000@10", "10+1000@100", "10+10000@100"]
SPARSITY_SETTINGS = [
    "1+10@10",
    "1+20@10",
    "1+40@10",
    "1+80@10",
    "1+160@10",
    "1+320@10",
    "1+640@10",
    "1+1280@10",
    "1+2560@10",
    "1+5120@10",
    "1+10240@10",
]
TOPK_SETTINGS = [f"1+1000@{k}" for k in [10, 20, 40, 60, 80, 100]]
BALANCED_RECEIVER_SETTINGS = ["10+1000@100", "10+10000@100"]

BASE_EVAL_ARGS = [
    "dataset=elliptic_recommendation",
    "experiment=exp_edge_recommendation",
    "experiment.tasks=[test]",
    "experiment.test.batch_size=16",
    "seed=0",
    "wandb.mode=offline",
]


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def tag(text: str) -> str:
    return text.replace("+", "p").replace("@", "at").replace(".", "p")


def print_command(cmd: list[str]) -> None:
    print("$", shlex.join(cmd))


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print_command(cmd)
    if DRY_RUN:
        return
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def run_streamed(cmd: list[str], log_path: Path | None = None) -> int:
    print_command(cmd)
    if DRY_RUN:
        if log_path is not None:
            print(f"[dry-run] would tee output to {rel(log_path)}")
        return 0

    if log_path is None:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return proc.returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return proc.wait()


def run_logged(cmd: list[str], log_path: Path) -> None:
    rc = run_streamed(cmd, log_path)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def append_line(path: Path, text: str) -> None:
    print(text)
    if DRY_RUN:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def child_max_memory_kb() -> int:
    try:
        import resource
    except ImportError:
        return 0
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def run_logged_timed(cmd: list[str], log_path: Path) -> None:
    start = time.perf_counter()
    rc = run_streamed(cmd, log_path)
    elapsed = time.perf_counter() - start
    append_line(log_path, f"elapsed_sec={elapsed:.3f}")
    append_line(log_path, f"max_mem_kb={child_max_memory_kb()}")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def main_cmd(name: str, args: Iterable[str]) -> list[str]:
    return [sys.executable, "-m", "main", f"+name={name}", *args]


def algorithm_args(
    algorithm: str = "iterative_filtering",
    *,
    anchor: bool = False,
    anchor_mode: str | None = None,
    anchor_fusion: str | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    args = [f"algorithm={algorithm}"]
    if anchor:
        args.append("algorithm.use_anchor_features=true")
    if anchor_mode is not None:
        args.append(f"algorithm.model.anchor_feature_mode={anchor_mode}")
    if anchor_fusion is not None:
        args.append(f"algorithm.model.anchor_fusion_mode={anchor_fusion}")
    args.extend(extra)
    return args


def eval_args(
    algorithm: str = "iterative_filtering",
    *,
    anchor: bool = False,
    anchor_mode: str | None = None,
    anchor_fusion: str | None = None,
    extra_algorithm_args: Iterable[str] = (),
    extra_args: Iterable[str] = (),
) -> list[str]:
    return [
        "dataset=elliptic_recommendation",
        *algorithm_args(
            algorithm,
            anchor=anchor,
            anchor_mode=anchor_mode,
            anchor_fusion=anchor_fusion,
            extra=extra_algorithm_args,
        ),
        *BASE_EVAL_ARGS[1:],
        *extra_args,
    ]


def train_args(
    *,
    seed: int,
    anchor: bool = False,
    anchor_mode: str | None = None,
    anchor_fusion: str | None = None,
    extra_algorithm_args: Iterable[str] = (),
    finetune: bool = False,
    load: str | None = None,
) -> list[str]:
    args = [
        "dataset=elliptic_recommendation",
        *algorithm_args(
            "iterative_filtering",
            anchor=anchor,
            anchor_mode=anchor_mode,
            anchor_fusion=anchor_fusion,
            extra=extra_algorithm_args,
        ),
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[training]",
        "experiment.validation.test_during_training=False",
        "wandb.mode=offline",
        f"seed={seed}",
    ]
    if finetune:
        args.extend(
            [
                "experiment.training.early_stopping.enabled=False",
                "experiment.training.max_epochs=300",
                "dataset.augment.enabled=True",
            ]
        )
    if load is not None:
        args.append(f"load={load}")
    return args


def latest_checkpoint() -> Path:
    outputs = PROJECT_ROOT / "outputs"
    checkpoints = [
        p
        for p in outputs.rglob("*.ckpt")
        if "checkpoints" in p.parts and p.is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found under outputs/**/checkpoints/*.ckpt")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def copy_latest_checkpoint(dest: str | Path) -> None:
    dest = PROJECT_ROOT / dest
    if DRY_RUN:
        print(f"[dry-run] would copy newest checkpoint to {rel(dest)}")
        return
    src = latest_checkpoint()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied {rel(src)} -> {rel(dest)}")


def print_file(path: str | Path) -> None:
    path = PROJECT_ROOT / path
    if DRY_RUN:
        print(f"[dry-run] would print {rel(path)}")
        return
    if path.exists():
        print(path.read_text(encoding="utf-8", errors="replace"))
    else:
        print(f"[WARN] missing {rel(path)}")


def summarize_revfilter(log_root: str | Path, prefix: str = "revfilter", *, show: bool = True) -> None:
    log_root = Path(log_root)
    cmd = python_cmd(str(PROJECT_ROOT / "scripts" / "summarize_revfilter_logs.py"), str(log_root))
    if prefix != "revfilter":
        cmd.extend(["--prefix", prefix])
    run(cmd)
    if show:
        print_file(log_root / f"{prefix}_summary.md")


def run_eval_grid(
    *,
    log_root: str,
    ckpts: Iterable[str],
    settings: Iterable[str],
    name_prefix: str,
    load_template: str,
    args_template: Callable[[str], list[str]],
    summarize_prefix: str | None = None,
    show_summary: bool = True,
) -> None:
    root = PROJECT_ROOT / log_root
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for ckpt in ckpts:
        for setting in settings:
            run_tag = tag(f"{ckpt}_{setting}")
            cmd = main_cmd(
                f"{name_prefix}_{run_tag}",
                [
                    *args_template(ckpt),
                    f"load={load_template.format(ckpt=ckpt)}",
                    f"+shortcut={setting}",
                ],
            )
            run_logged(cmd, root / f"{run_tag}.log")
    if summarize_prefix is not None:
        summarize_revfilter(root, prefix=summarize_prefix, show=show_summary)


def model_algorithm(model: str) -> str:
    return "lightgcn" if model == "LightGCN" else model.lower()


def run_baseline_models(
    *,
    log_suffix: str,
    settings: Iterable[str],
    name_middle: str,
    summarize_prefix: str = "revfilter",
) -> None:
    for model in ["MLP", "NGCF", "LightGCN"]:
        log_root = PROJECT_ROOT / f"logs-{model.lower()}-{log_suffix}"
        if not DRY_RUN:
            log_root.mkdir(parents=True, exist_ok=True)
        for ckpt in ["0", "1", "2"]:
            for setting in settings:
                run_tag = tag(f"{ckpt}_{setting}")
                run_name = f"{model}_{run_tag}" if not name_middle else f"{model}_{name_middle}_{run_tag}"
                cmd = main_cmd(
                    run_name,
                    [
                        *eval_args(model_algorithm(model)),
                        f"load=checkpoints/{model}/{ckpt}.ckpt",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, log_root / f"{run_tag}.log")
        summarize_revfilter(log_root, prefix=summarize_prefix)


def train_anchor_seed(
    seed: int,
    *,
    ckpt_dir: str = "checkpoints/AnchorRevFilter",
    name_prefix: str = "AnchorRevFilter",
    anchor_mode: str | None = None,
    anchor_fusion: str | None = None,
    extra_algorithm_args: Iterable[str] = (),
    pretrain_name: str | None = None,
    tuned_name: str | None = None,
) -> None:
    pretrain_name = pretrain_name or f"pretrain_seed{seed}.ckpt"
    tuned_name = tuned_name or f"tuned_seed{seed}.ckpt"
    ckpt_root = PROJECT_ROOT / ckpt_dir
    if not DRY_RUN:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    pretrain_cmd = main_cmd(
        f"{name_prefix}_pretrain_seed{seed}",
        train_args(
            seed=seed,
            anchor=True,
            anchor_mode=anchor_mode,
            anchor_fusion=anchor_fusion,
            extra_algorithm_args=extra_algorithm_args,
        ),
    )
    run(pretrain_cmd)
    copy_latest_checkpoint(Path(ckpt_dir) / pretrain_name)

    pretrain_load = str(Path(ckpt_dir) / pretrain_name).replace("\\", "/")
    finetune_cmd = main_cmd(
        f"{name_prefix}_finetune_seed{seed}",
        train_args(
            seed=seed,
            anchor=True,
            anchor_mode=anchor_mode,
            anchor_fusion=anchor_fusion,
            extra_algorithm_args=extra_algorithm_args,
            finetune=True,
            load=pretrain_load,
        ),
    )
    run(finetune_cmd)
    copy_latest_checkpoint(Path(ckpt_dir) / tuned_name)


def train_anchor_modes(modes: Iterable[str], seeds: Iterable[int]) -> None:
    for mode in modes:
        for seed in seeds:
            train_anchor_seed(
                seed,
                ckpt_dir="checkpoints/AnchorRevFilter",
                name_prefix=f"AnchorAblation_{mode}",
                anchor_mode=mode,
                pretrain_name=f"{mode}_pretrain_seed{seed}.ckpt",
                tuned_name=f"{mode}_tuned_seed{seed}.ckpt",
            )


def task_anchor_only() -> None:
    for seed in [0, 1, 2]:
        train_anchor_seed(
            seed,
            ckpt_dir="checkpoints/SAIFAnchorOnly",
            name_prefix="SAIF_anchor_only",
            anchor_mode="full",
            anchor_fusion="anchor_only",
        )


def task_train_anchor_seed1() -> None:
    train_anchor_seed(1)


def task_train_anchor_seed2() -> None:
    train_anchor_seed(2)


def task_train_anchor_seeds_3_4() -> None:
    for seed in [3, 4]:
        train_anchor_seed(seed, anchor_mode="full")


def task_check_checkpoints() -> None:
    if DRY_RUN:
        print("[dry-run] would inspect selected checkpoint tensor shapes")
        return
    import torch

    paths = [
        "checkpoints/SAIFAnchorOnly/tuned_seed0.ckpt",
        "checkpoints/AnchorRevFilter/tuned_seed0.ckpt",
        "checkpoints/AnchorRevFilter/size_only_tuned_seed0.ckpt",
    ]
    keys = ["model.anchor_norm.weight", "model.pred_mlp.1.weight"]
    for path in paths:
        print("=" * 80)
        print(path)
        try:
            ckpt = torch.load(PROJECT_ROOT / path, map_location="cpu")
        except FileNotFoundError:
            print("NOT FOUND")
            continue
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        for key in keys:
            print(key, tuple(state_dict[key].shape) if key in state_dict else "MISSING")


def parse_mean(text: str) -> float:
    return float(str(text).split("+/-")[0])


def load_curve_rows(path: str | Path, mode: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with (PROJECT_ROOT / path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            setting = row["setting"]
            item: dict[str, float | str] = {
                "setting": setting,
                "hr": parse_mean(row["HR"]),
                "ndcg": parse_mean(row["NDCG"]),
            }
            if mode == "topk":
                item["top_k"] = int(setting.split("@")[-1])
            else:
                item["density"] = float(row["density"]) * 100.0
            rows.append(item)
    key = "top_k" if mode == "topk" else "density"
    rows.sort(key=lambda r: float(r[key]), reverse=(mode == "sparsity"))
    return rows


def plot_curve(mode: str, metric: str) -> None:
    if DRY_RUN:
        print(f"[dry-run] would draw {mode} {metric} curve")
        return
    import matplotlib.pyplot as plt

    if mode == "topk":
        methods = {
            "MLP": "logs-mlp-topk/revfilter_summary.csv",
            "NGCF": "logs-ngcf-topk/revfilter_summary.csv",
            "LightGCN": "logs-lightgcn-topk/revfilter_summary.csv",
            "RevFilter": "logs-official-topk/revfilter_summary.csv",
            "AnchorRevFilter": "logs-anchor-topk/revfilter_summary.csv",
        }
    else:
        methods = {
            "MLP": "logs-mlp-sparsity/revfilter_summary.csv",
            "NGCF": "logs-ngcf-sparsity/revfilter_summary.csv",
            "LightGCN": "logs-lightgcn-sparsity/revfilter_summary.csv",
            "RevFilter": "logs-official-sparsity/revfilter_summary.csv",
            "AnchorRevFilter": "logs-anchor-sparsity/revfilter_summary.csv",
        }

    styles = {
        "MLP": dict(color="#9ca3af", marker="o", linestyle=":"),
        "NGCF": dict(color="#6b7280", marker="s", linestyle=":"),
        "LightGCN": dict(color="#4b5563", marker="^", linestyle="--"),
        "RevFilter": dict(color="#2563eb", marker="D", linestyle="-"),
        "AnchorRevFilter": dict(
            color="#dc2626",
            marker="*",
            linestyle="-",
            linewidth=2.5,
            markersize=10,
        ),
    }

    out_dir = PROJECT_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))

    for method, path in methods.items():
        rows = load_curve_rows(path, mode)
        x_key = "top_k" if mode == "topk" else "density"
        x = [r[x_key] for r in rows]
        y = [r[metric] for r in rows]
        plt.plot(x, y, label=method, **styles[method])

    if mode == "topk":
        plt.xlabel("Top-k budget")
        plt.xticks([10, 20, 40, 60, 80, 100])
        out_name = f"topk_{metric}_curve"
    else:
        plt.xscale("log")
        plt.gca().invert_xaxis()
        plt.xlabel("Candidate density (%)")
        out_name = f"sparsity_{metric}_curve"

    plt.ylabel(metric.upper())
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{out_name}.pdf")
    plt.savefig(out_dir / f"{out_name}.png", dpi=300)
    print(f"Wrote {rel(out_dir / f'{out_name}.pdf')}")
    print(f"Wrote {rel(out_dir / f'{out_name}.png')}")


def task_draw_topk_hr() -> None:
    plot_curve("topk", "hr")


def task_draw_topk_ndcg() -> None:
    plot_curve("topk", "ndcg")


def task_draw_sparsity_hr() -> None:
    plot_curve("sparsity", "hr")


def task_draw_sparsity_ndcg() -> None:
    plot_curve("sparsity", "ndcg")


def task_draw_case_study_43() -> None:
    if DRY_RUN:
        print("[dry-run] would draw figures/case43_hit_ranks.{pdf,png}")
        return
    import matplotlib.pyplot as plt

    rev_ranks = [12, 13, 16, 18, 22, 45, 52, 67, 69, 74]
    anchor_ranks = [1, 2, 3, 4, 5, 11, 19, 48, 64, 67]
    out_dir = PROJECT_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7.2, 2.2))
    plt.scatter(rev_ranks, [1] * len(rev_ranks), s=70, marker="o", color="#2563eb", label="RevFilter")
    plt.scatter(
        anchor_ranks,
        [0] * len(anchor_ranks),
        s=90,
        marker="*",
        color="#dc2626",
        label="AnchorRevFilter",
    )
    plt.yticks([1, 0], ["RevFilter", "AnchorRevFilter"])
    plt.xlim(0, 101)
    plt.xticks([1, 10, 20, 40, 60, 80, 100])
    plt.xlabel("Rank position of illicit edges (lower is better)")
    plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    plt.title("Case study on 10+1000@100: same HR, better ranking")
    plt.tight_layout()
    plt.savefig(out_dir / "case43_hit_ranks.pdf")
    plt.savefig(out_dir / "case43_hit_ranks.png", dpi=300)
    print("Wrote figures/case43_hit_ranks.pdf")
    print("Wrote figures/case43_hit_ranks.png")


def task_evaluate_anchor_only() -> None:
    run_eval_grid(
        log_root="logs-saif-anchor-only-table2",
        ckpts=["tuned_seed0", "tuned_seed1", "tuned_seed2"],
        settings=TABLE2_SETTINGS,
        name_prefix="SAIF_anchor_only",
        load_template="checkpoints/SAIFAnchorOnly/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(anchor=True, anchor_mode="full", anchor_fusion="anchor_only"),
        summarize_prefix="revfilter",
    )


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def task_summarize_cost() -> None:
    root = PROJECT_ROOT / "logs-cost"
    rows = []
    for path in sorted(root.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        name = path.stem
        method = "Anchor" if name.startswith("anchor") else "Official"
        setting = name.split("_", 1)[1].replace("p", "+").replace("at", "@")
        rows.append(
            {
                "method": method,
                "setting": setting,
                "HR": re.findall(r"final_test/HR\D+([0-9.]+)", text)[-1]
                if re.findall(r"final_test/HR\D+([0-9.]+)", text)
                else "",
                "NDCG": re.findall(r"final_test/NDCG\D+([0-9.]+)", text)[-1]
                if re.findall(r"final_test/NDCG\D+([0-9.]+)", text)
                else "",
                "elapsed_sec": re.findall(r"elapsed_sec=([0-9.]+)", text)[-1]
                if re.findall(r"elapsed_sec=([0-9.]+)", text)
                else "",
                "max_mem_kb": re.findall(r"max_mem_kb=([0-9]+)", text)[-1]
                if re.findall(r"max_mem_kb=([0-9]+)", text)
                else "",
            }
        )
    out = root / "cost_summary.md"
    if not DRY_RUN:
        with out.open("w", encoding="utf-8") as f:
            f.write("| method | setting | HR | NDCG | elapsed_sec | max_mem_kb |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['method']} | {row['setting']} | {row['HR']} | {row['NDCG']} | "
                    f"{row['elapsed_sec']} | {row['max_mem_kb']} |\n"
                )
    print(f"Wrote: {rel(out)}")
    print_file(out)


def task_summarize_balanced_receiver_control() -> None:
    root = PROJECT_ROOT / "logs-balanced-receiver-control"
    rows = []
    for path in sorted(root.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.match(r"(official|saif|geometry)_ckpt(\d+)_(\d+)p(\d+)at(\d+)$", path.stem)
        if not match:
            print(f"[WARN] skip {path.stem}")
            continue
        method, ckpt, n_pos, n_neg, top_k = match.groups()
        rows.append(
            {
                "method": method,
                "ckpt": ckpt,
                "setting": f"{n_pos}+{n_neg}@{top_k}",
                "density": last_float(r"Avg density:\s*([0-9.]+)", text),
                "HR": last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?)", text),
                "NDCG": last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?)", text),
                "log_path": str(path),
            }
        )

    raw_path = root / "balanced_receiver_control_raw.csv"
    md_path = root / "balanced_receiver_control_summary.md"
    if not DRY_RUN:
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "ckpt", "setting", "density", "HR", "NDCG", "log_path"],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_rows = []
        for setting in sorted({r["setting"] for r in rows}):
            for method in ["official", "geometry", "saif"]:
                group = [r for r in rows if r["setting"] == setting and r["method"] == method]
                if not group:
                    continue

                def mean_std(metric: str) -> str:
                    values = [r[metric] for r in group if r[metric] is not None]
                    if not values:
                        return ""
                    mean = statistics.mean(values)
                    std = statistics.stdev(values) if len(values) > 1 else 0.0
                    return f"{mean:.6f}+/-{std:.6f}"

                density = group[0]["density"]
                summary_rows.append(
                    {
                        "setting": setting,
                        "method": method,
                        "n": len(group),
                        "density": "" if density is None else f"{density:.8f}",
                        "HR": mean_std("HR"),
                        "NDCG": mean_std("NDCG"),
                    }
                )

        with md_path.open("w", encoding="utf-8") as f:
            f.write("| setting | method | n | density | HR | NDCG |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for row in summary_rows:
                f.write(
                    f"| {row['setting']} | {row['method']} | {row['n']} | {row['density']} | "
                    f"{row['HR']} | {row['NDCG']} |\n"
                )

    print(f"Wrote: {rel(raw_path)}")
    print(f"Wrote: {rel(md_path)}")
    print_file(md_path)


def task_keep_multiplier() -> None:
    root = PROJECT_ROOT / "logs-keep-ratio-sensitivity"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for km in ["1.0", "1.25", "1.5", "2.0", "3.0"]:
        km_tag = tag(km)
        for ckpt in ["0_tuned", "1_tuned", "2_tuned"]:
            for setting in ["10+1000@100", "10+10000@100"]:
                run_tag = tag(setting)
                cmd = main_cmd(
                    f"RevFilter_km{km_tag}_{ckpt}_{run_tag}",
                    [
                        *eval_args(extra_algorithm_args=[f"algorithm.keep_multiplier={km}"]),
                        f"load=checkpoints/RevTrack/{ckpt}.ckpt",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, root / f"official_km{km_tag}_{ckpt}_{run_tag}.log")
        for ckpt in ["tuned_seed0", "tuned_seed1", "tuned_seed2"]:
            for setting in ["10+1000@100", "10+10000@100"]:
                run_tag = tag(setting)
                cmd = main_cmd(
                    f"SAIF_km{km_tag}_{ckpt}_{run_tag}",
                    [
                        *eval_args(
                            anchor=True,
                            anchor_mode="full",
                            anchor_fusion="full",
                            extra_algorithm_args=[f"algorithm.keep_multiplier={km}"],
                        ),
                        f"load=checkpoints/AnchorRevFilter/{ckpt}.ckpt",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, root / f"saif_km{km_tag}_{ckpt}_{run_tag}.log")
    run(
        python_cmd(
            str(PROJECT_ROOT / "scripts" / "summarize_keep_sensitivity.py"),
            str(root),
            "--baseline",
            "official",
            "--method",
            "saif",
        )
    )
    print_file(root / "keep_sensitivity_delta.md")


def task_receiver_balance_control() -> None:
    root = PROJECT_ROOT / "logs-balanced-receiver-control"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for ckpt_id in ["0", "1", "2"]:
        for setting in ["10+1000@100", "10+10000@100"]:
            run_tag = tag(setting)
            cmd = main_cmd(
                f"Bal_Geometry_ckpt{ckpt_id}_{run_tag}",
                [
                    *eval_args(
                        anchor=True,
                        anchor_mode="size_only",
                        extra_args=["dataset.eval_pool_mode=balanced_receivers"],
                    ),
                    f"load=checkpoints/AnchorRevFilter/size_only_tuned_seed{ckpt_id}.ckpt",
                    f"+shortcut={setting}",
                ],
            )
            run_logged(cmd, root / f"geometry_ckpt{ckpt_id}_{run_tag}.log")


def task_run_3models_topk() -> None:
    run_baseline_models(log_suffix="topk", settings=TOPK_SETTINGS, name_middle="TopK")


def task_run_3models_sparsity() -> None:
    run_baseline_models(log_suffix="sparsity", settings=SPARSITY_SETTINGS, name_middle="Sparsity")


def task_run_baselines_table2() -> None:
    run_baseline_models(log_suffix="table2", settings=TABLE2_SETTINGS, name_middle="")


def task_run_anchor() -> None:
    root = PROJECT_ROOT / "logs-anchor-revfilter"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for setting in MAIN_SETTINGS:
        run_tag = tag(setting)
        cmd = main_cmd(
            f"AnchorRevFilter_{run_tag}",
            [
                *eval_args(anchor=True),
                "load=checkpoints/AnchorRevFilter/tuned_seed0.ckpt",
                f"+shortcut={setting}",
            ],
        )
        run_logged(cmd, root / f"{run_tag}.log")
    summarize_revfilter(root, prefix="official_revfilter")


def task_run_anchor_cost() -> None:
    root = PROJECT_ROOT / "logs-cost"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for setting in ["10+1000@100", "10+10000@100"]:
        run_tag = tag(setting)
        cmd = main_cmd(
            f"AnchorCost_anchor_{run_tag}",
            [
                *eval_args(anchor=True, anchor_mode="full"),
                "load=checkpoints/AnchorRevFilter/tuned_seed0.ckpt",
                f"+shortcut={setting}",
            ],
        )
        run_logged_timed(cmd, root / f"anchor_{run_tag}.log")


def task_run_official_cost() -> None:
    root = PROJECT_ROOT / "logs-cost"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for setting in ["10+1000@100", "10+10000@100"]:
        run_tag = tag(setting)
        cmd = main_cmd(
            f"OfficialCost_official_{run_tag}",
            [
                *eval_args(),
                "load=checkpoints/RevTrack/0_tuned.ckpt",
                f"+shortcut={setting}",
            ],
        )
        run_logged_timed(cmd, root / f"official_{run_tag}.log")


def task_run_complexity_profile() -> None:
    root = PROJECT_ROOT / "logs-complexity-profile"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    settings = ["10+1000@100", "10+10000@100"]
    profile_arg = "algorithm.profile_search=true"

    for ckpt_id in ["0", "1", "2"]:
        for setting in settings:
            run_tag = tag(setting)
            jobs = [
                (
                    "official",
                    f"OfficialComplexity_ckpt{ckpt_id}_{run_tag}",
                    eval_args(extra_algorithm_args=[profile_arg]),
                    f"checkpoints/RevTrack/{ckpt_id}_tuned.ckpt",
                ),
                (
                    "saif",
                    f"SAIFComplexity_ckpt{ckpt_id}_{run_tag}",
                    eval_args(
                        anchor=True,
                        anchor_mode="full",
                        extra_algorithm_args=[profile_arg],
                    ),
                    f"checkpoints/AnchorRevFilter/tuned_seed{ckpt_id}.ckpt",
                ),
            ]
            for method, run_name, args, load in jobs:
                cmd = main_cmd(
                    run_name,
                    [*args, f"load={load}", f"+shortcut={setting}"],
                )
                run_logged_timed(
                    cmd, root / f"{method}_ckpt{ckpt_id}_{run_tag}.log"
                )

    task_summarize_complexity_profile()


def task_summarize_complexity_profile() -> None:
    run(
        python_cmd(
            str(PROJECT_ROOT / "scripts" / "summarize_complexity_profile.py"),
            str(PROJECT_ROOT / "logs-complexity-profile"),
        )
    )


def task_run_anchor_result() -> None:
    run_eval_grid(
        log_root="logs-anchor-revfilter-3ckpts",
        ckpts=["tuned_seed0", "tuned_seed1", "tuned_seed2"],
        settings=MAIN_SETTINGS,
        name_prefix="AnchorRevFilter",
        load_template="checkpoints/AnchorRevFilter/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(anchor=True),
        summarize_prefix="revfilter",
    )


def task_run_anchor_sparsity() -> None:
    run_eval_grid(
        log_root="logs-anchor-sparsity",
        ckpts=["tuned_seed0", "tuned_seed1", "tuned_seed2"],
        settings=SPARSITY_SETTINGS,
        name_prefix="AnchorSparsity",
        load_template="checkpoints/AnchorRevFilter/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(anchor=True, anchor_mode="full"),
        summarize_prefix="revfilter",
        show_summary=False,
    )
    summarize_revfilter("logs-anchor-sparsity", prefix="sparsity", show=False)


def task_run_anchor_topk() -> None:
    run_eval_grid(
        log_root="logs-anchor-topk",
        ckpts=["tuned_seed0", "tuned_seed1", "tuned_seed2"],
        settings=TOPK_SETTINGS,
        name_prefix="AnchorTopK",
        load_template="checkpoints/AnchorRevFilter/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(anchor=True, anchor_mode="full"),
        summarize_prefix="revfilter",
        show_summary=False,
    )


def task_run_case_study() -> None:
    run(
        python_cmd(
            str(PROJECT_ROOT / "scripts" / "case_study_anchor_revfilter.py"),
            "--setting",
            "10+1000@100",
            "--official-ckpt",
            "checkpoints/RevTrack/0_tuned.ckpt",
            "--anchor-ckpt",
            "checkpoints/AnchorRevFilter/tuned_seed0.ckpt",
            "--seed",
            "0",
            "--top-cases",
            "5",
            "--out",
            "logs-case-study",
        )
    )


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["setting"]: row for row in csv.DictReader(f)}


def summary_mean(text: str) -> float:
    return float(text.split("+/-")[0])


def compare_summaries(
    official_path: Path,
    anchor_path: Path,
    out_path: Path,
    *,
    include_top_k: bool = False,
    include_density: bool = False,
) -> None:
    official = read_summary(official_path)
    anchor = read_summary(anchor_path)
    settings = sorted(anchor.keys())
    if include_top_k:
        settings = sorted(settings, key=lambda s: int(s.split("@")[1]))
    if include_density:
        settings = sorted(settings, key=lambda s: int(s.split("+")[1].split("@")[0]))
    if not DRY_RUN:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            if include_top_k:
                f.write("| setting | top_k | Official HR | Anchor HR | delta HR | Official NDCG | Anchor NDCG | delta NDCG |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            elif include_density:
                f.write("| setting | density | Official HR | Anchor HR | delta HR | Official NDCG | Anchor NDCG | delta NDCG |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            else:
                f.write("| setting | Official HR | Anchor HR | delta HR | Official NDCG | Anchor NDCG | delta NDCG |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for setting in settings:
                official_row = official[setting]
                anchor_row = anchor[setting]
                official_hr = summary_mean(official_row["HR"])
                anchor_hr = summary_mean(anchor_row["HR"])
                official_ndcg = summary_mean(official_row["NDCG"])
                anchor_ndcg = summary_mean(anchor_row["NDCG"])
                prefix = f"| {setting} | "
                if include_top_k:
                    prefix += f"{setting.split('@')[1]} | "
                elif include_density:
                    prefix += f"{anchor_row['density']} | "
                f.write(
                    f"{prefix}{official_hr:.6f} | {anchor_hr:.6f} | {anchor_hr - official_hr:+.6f} | "
                    f"{official_ndcg:.6f} | {anchor_ndcg:.6f} | {anchor_ndcg - official_ndcg:+.6f} |\n"
                )
    print(f"Wrote: {rel(out_path)}")
    print_file(out_path)


def task_compare_anchor_official_table2() -> None:
    compare_summaries(
        PROJECT_ROOT / "logs-official-revfilter-table2" / "revfilter_summary.csv",
        PROJECT_ROOT / "logs-anchor-revfilter-table2" / "revfilter_summary.csv",
        PROJECT_ROOT / "logs-anchor-revfilter-table2" / "anchor_vs_official_delta.md",
    )


def first_existing(*paths: str) -> Path:
    for path in paths:
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / paths[0]


def task_compare_anchor_official_sparsity() -> None:
    compare_summaries(
        first_existing(
            "logs-official-sparsity/sparsity_summary.csv",
            "logs-official-sparsity/revfilter_summary.csv",
        ),
        first_existing(
            "logs-anchor-sparsity/sparsity_summary.csv",
            "logs-anchor-sparsity/revfilter_summary.csv",
        ),
        PROJECT_ROOT / "logs-anchor-sparsity" / "anchor_vs_official_sparsity_delta.md",
        include_density=True,
    )


def task_compare_anchor_official_topk() -> None:
    compare_summaries(
        PROJECT_ROOT / "logs-official-topk" / "revfilter_summary.csv",
        PROJECT_ROOT / "logs-anchor-topk" / "revfilter_summary.csv",
        PROJECT_ROOT / "logs-anchor-topk" / "anchor_vs_official_topk_delta.md",
        include_top_k=True,
    )


def task_paired_official_anchor() -> None:
    run(
        python_cmd(
            str(PROJECT_ROOT / "scripts" / "paired_revfilter_tests.py"),
            "--a",
            "logs-official-revfilter-table2/revfilter_raw.csv",
            "--b",
            "logs-anchor-revfilter-table2/revfilter_raw.csv",
            "--a-name",
            "Official",
            "--b-name",
            "Anchor",
            "--out",
            "logs-anchor-revfilter-table2/paired_official_anchor.md",
        )
    )
    print_file("logs-anchor-revfilter-table2/paired_official_anchor.md")


def export_receiver_balanced_instance_metrics(
    *,
    method: str,
    ckpts: Iterable[tuple[str, str]],
    out: str,
    overrides: Iterable[str] = (),
) -> None:
    cmd = python_cmd(
        str(PROJECT_ROOT / "scripts" / "export_edge_instance_metrics.py"),
        "--method",
        method,
        "--algorithm",
        "iterative_filtering",
        "--settings",
        *BALANCED_RECEIVER_SETTINGS,
        "--out",
        out,
        "--seed",
        "0",
        "--batch-size",
        "16",
    )
    for ckpt_id, ckpt_path in ckpts:
        cmd.extend(["--ckpt", ckpt_path, "--ckpt-id", ckpt_id])
    cmd.extend(["dataset.eval_pool_mode=balanced_receivers", *overrides])
    run(cmd)


def run_instance_wilcoxon(
    *,
    a: str,
    b: str,
    a_name: str,
    b_name: str,
    out: str,
) -> None:
    run(
        python_cmd(
            str(PROJECT_ROOT / "scripts" / "paired_instance_wilcoxon_bh.py"),
            "--a",
            a,
            "--b",
            b,
            "--a-name",
            a_name,
            "--b-name",
            b_name,
            "--out",
            out,
            "--alternative",
            "two-sided",
            "--bh-scope",
            "all",
        )
    )


def task_receiver_balanced_instance_wilcoxon() -> None:
    root = PROJECT_ROOT / "logs-balanced-receiver-instance"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)

    official_csv = str(root / "official_instance_metrics.csv")
    saif_csv = str(root / "saif_instance_metrics.csv")
    geometry_csv = str(root / "geometry_instance_metrics.csv")

    export_receiver_balanced_instance_metrics(
        method="official",
        ckpts=[
            ("0", "checkpoints/RevTrack/0_tuned.ckpt"),
            ("1", "checkpoints/RevTrack/1_tuned.ckpt"),
            ("2", "checkpoints/RevTrack/2_tuned.ckpt"),
        ],
        out=official_csv,
    )
    export_receiver_balanced_instance_metrics(
        method="saif",
        ckpts=[
            ("0", "checkpoints/AnchorRevFilter/tuned_seed0.ckpt"),
            ("1", "checkpoints/AnchorRevFilter/tuned_seed1.ckpt"),
            ("2", "checkpoints/AnchorRevFilter/tuned_seed2.ckpt"),
        ],
        out=saif_csv,
        overrides=["algorithm.use_anchor_features=true"],
    )
    export_receiver_balanced_instance_metrics(
        method="geometry",
        ckpts=[
            ("0", "checkpoints/AnchorRevFilter/size_only_tuned_seed0.ckpt"),
            ("1", "checkpoints/AnchorRevFilter/size_only_tuned_seed1.ckpt"),
            ("2", "checkpoints/AnchorRevFilter/size_only_tuned_seed2.ckpt"),
        ],
        out=geometry_csv,
        overrides=[
            "algorithm.use_anchor_features=true",
            "algorithm.model.anchor_feature_mode=size_only",
        ],
    )

    run_instance_wilcoxon(
        a=official_csv,
        b=saif_csv,
        a_name="Official",
        b_name="SAIF",
        out=str(root / "paired_official_saif.csv"),
    )
    run_instance_wilcoxon(
        a=official_csv,
        b=geometry_csv,
        a_name="Official",
        b_name="Geometry",
        out=str(root / "paired_official_geometry.csv"),
    )

    print_file(root / "paired_official_saif.md")
    print_file(root / "paired_official_geometry.md")


def task_build_edge_index() -> None:
    if DRY_RUN:
        print("[dry-run] would instantiate EllipticRecommendationDataset and build edge_index.pt")
        return
    from omegaconf import OmegaConf
    from datasets.elliptic.dataset import EllipticRecommendationDataset

    cfg = OmegaConf.create(
        {
            "shot_size": -1,
            "num_illicits": 5,
            "num_licits": 10,
            "num_samples": 256,
            "filter_1_1": True,
            "augment": {"enabled": False, "min": 1, "max": 20, "gamma": 0.4},
            "use_edge_index": True,
        }
    )
    EllipticRecommendationDataset(cfg)
    edge_index = PROJECT_ROOT / "data" / "elliptic" / "processed" / "edge_index.pt"
    if edge_index.exists():
        print(f"{rel(edge_index)} {edge_index.stat().st_size} bytes")


def task_run_official_result() -> None:
    run_eval_grid(
        log_root="logs-official-revfilter",
        ckpts=["0_tuned", "1_tuned", "2_tuned"],
        settings=MAIN_SETTINGS,
        name_prefix="Official_RevFilter",
        load_template="checkpoints/RevTrack/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(),
        summarize_prefix="official_revfilter",
    )


def task_run_official_sparsity() -> None:
    run_eval_grid(
        log_root="logs-official-sparsity",
        ckpts=["0_tuned", "1_tuned", "2_tuned"],
        settings=SPARSITY_SETTINGS,
        name_prefix="OfficialSparsity",
        load_template="checkpoints/RevTrack/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(),
        summarize_prefix="revfilter",
        show_summary=False,
    )
    summarize_revfilter("logs-official-sparsity", prefix="sparsity", show=False)


def task_run_official_topk() -> None:
    run_eval_grid(
        log_root="logs-official-topk",
        ckpts=["0_tuned", "1_tuned", "2_tuned"],
        settings=TOPK_SETTINGS,
        name_prefix="OfficialTopK",
        load_template="checkpoints/RevTrack/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(),
        summarize_prefix="revfilter",
        show_summary=False,
    )


def task_run_anchor_table2() -> None:
    run_eval_grid(
        log_root="logs-anchor-revfilter-table2",
        ckpts=["tuned_seed0", "tuned_seed1", "tuned_seed2"],
        settings=TABLE2_SETTINGS,
        name_prefix="AnchorRevFilter",
        load_template="checkpoints/AnchorRevFilter/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(anchor=True),
        summarize_prefix="revfilter",
    )


def task_run_official_table2() -> None:
    run_eval_grid(
        log_root="logs-official-revfilter-table2",
        ckpts=["0_tuned", "1_tuned", "2_tuned"],
        settings=TABLE2_SETTINGS,
        name_prefix="Official_RevFilter",
        load_template="checkpoints/RevTrack/{ckpt}.ckpt",
        args_template=lambda _ckpt: eval_args(),
        summarize_prefix="revfilter",
    )


def summarize_anchor_ablation(root: Path, *, has_seed: bool) -> None:
    rows = []
    pattern = (
        r"(.+?)_seed(\d+)_(\d+)p(\d+)at(\d+)$"
        if has_seed
        else r"(.+?)_(\d+)p(\d+)at(\d+)$"
    )
    for log_path in sorted(root.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.match(pattern, log_path.stem)
        if not match:
            print(f"[WARN] cannot parse name: {log_path.stem}")
            continue
        if has_seed:
            mode, seed, n_pos, n_neg, top_k = match.groups()
        else:
            mode, n_pos, n_neg, top_k = match.groups()
            seed = ""
        density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
        row = {
            "mode": mode,
            "seed": seed,
            "setting": f"{n_pos}+{n_neg}@{top_k}",
            "density": "" if density is None else f"{density:.8f}",
            "sparsity": "" if density is None else f"{1 - density:.8f}",
            "HR": value_or_empty(last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)),
            "NDCG": value_or_empty(last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)),
            "log_path": str(log_path),
        }
        rows.append(row)

    raw_path = root / "anchor_ablation_raw.csv"
    md_path = root / "anchor_ablation_summary.md"
    if DRY_RUN:
        print(f"[dry-run] would write {rel(raw_path)} and {rel(md_path)}")
        return
    fieldnames = ["mode", "setting", "density", "sparsity", "HR", "NDCG", "log_path"]
    if has_seed:
        fieldnames.insert(1, "seed")
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if has_seed:
        summary_rows = []
        for setting in sorted({r["setting"] for r in rows}):
            for mode in sorted({r["mode"] for r in rows if r["setting"] == setting}):
                group = [r for r in rows if r["setting"] == setting and r["mode"] == mode]

                def mean_std(metric: str) -> str:
                    values = [float(r[metric]) for r in group if r[metric] != ""]
                    if not values:
                        return ""
                    mean = statistics.mean(values)
                    std = statistics.stdev(values) if len(values) > 1 else 0.0
                    return f"{mean:.6f}+/-{std:.6f}"

                summary_rows.append(
                    {
                        "setting": setting,
                        "mode": mode,
                        "n": len(group),
                        "density": group[0]["density"],
                        "sparsity": group[0]["sparsity"],
                        "HR": mean_std("HR"),
                        "NDCG": mean_std("NDCG"),
                    }
                )
        with md_path.open("w", encoding="utf-8") as f:
            f.write("| setting | mode | n | density | sparsity | HR | NDCG |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|\n")
            for row in summary_rows:
                f.write(
                    f"| {row['setting']} | {row['mode']} | {row['n']} | {row['density']} | "
                    f"{row['sparsity']} | {row['HR']} | {row['NDCG']} |\n"
                )
    else:
        with md_path.open("w", encoding="utf-8") as f:
            f.write("| setting | mode | density | sparsity | HR | NDCG |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for row in sorted(rows, key=lambda x: (x["setting"], x["mode"])):
                f.write(
                    f"| {row['setting']} | {row['mode']} | {row['density']} | {row['sparsity']} | "
                    f"{row['HR']} | {row['NDCG']} |\n"
                )
    print(f"Wrote: {rel(raw_path)}")
    print(f"Wrote: {rel(md_path)}")
    print_file(md_path)


def value_or_empty(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def run_named_anchor_ablation(
    root: Path,
    *,
    jobs: Iterable[dict[str, str | list[str]]],
    settings: Iterable[str],
) -> None:
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        mode = str(job["mode"])
        anchor_mode = str(job.get("anchor_mode", "full"))
        load_template = str(job["load_template"])
        extra_algorithm_args = job.get("extra_algorithm_args", [])
        for seed in [0, 1, 2]:
            for setting in settings:
                run_tag = tag(setting)
                load = load_template.format(seed=seed)
                cmd = main_cmd(
                    f"AnchorAblation_{tag(f'{mode}_seed{seed}_{setting}')}",
                    [
                        *eval_args(
                            anchor=True,
                            anchor_mode=anchor_mode,
                            extra_algorithm_args=extra_algorithm_args,
                        ),
                        f"load={load}",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, root / f"{mode}_seed{seed}_{run_tag}.log")


def task_run_finetune_ablation() -> None:
    root = PROJECT_ROOT / "logs-finetune-ablation"
    settings = ["10+1000@100", "10+10000@100"]
    jobs = [
        {
            "mode": "full_tuned",
            "anchor_mode": "full",
            "load_template": "checkpoints/AnchorRevFilter/tuned_seed{seed}.ckpt",
        },
        {
            "mode": "no_finetune",
            "anchor_mode": "full",
            "load_template": "checkpoints/AnchorRevFilter/pretrain_seed{seed}.ckpt",
        },
    ]
    run_named_anchor_ablation(root, jobs=jobs, settings=settings)
    summarize_anchor_ablation(root, has_seed=True)


def task_run_targeted_ablation() -> None:
    root = PROJECT_ROOT / "logs-targeted-ablation"
    settings = ["10+1000@100", "10+10000@100"]
    jobs = [
        {
            "mode": "full",
            "anchor_mode": "full",
            "load_template": "checkpoints/AnchorRevFilter/tuned_seed{seed}.ckpt",
        },
        {
            "mode": "no_finetune",
            "anchor_mode": "full",
            "load_template": "checkpoints/AnchorRevFilter/pretrain_seed{seed}.ckpt",
        },
        {
            "mode": "size_only",
            "anchor_mode": "size_only",
            "load_template": "checkpoints/AnchorRevFilter/size_only_tuned_seed{seed}.ckpt",
        },
        {
            "mode": "no_density",
            "anchor_mode": "no_density",
            "load_template": "checkpoints/AnchorRevFilter/no_density_tuned_seed{seed}.ckpt",
        },
        {
            "mode": "no_balance",
            "anchor_mode": "no_balance",
            "load_template": "checkpoints/AnchorRevFilter/no_balance_tuned_seed{seed}.ckpt",
        },
    ]
    run_named_anchor_ablation(root, jobs=jobs, settings=settings)
    summarize_anchor_ablation(root, has_seed=True)


def task_train_no_layernorm() -> None:
    for seed in [0, 1, 2]:
        train_anchor_seed(
            seed,
            ckpt_dir="checkpoints/AnchorRevFilter",
            name_prefix="AnchorAblation_no_layernorm",
            anchor_mode="full",
            extra_algorithm_args=["algorithm.model.anchor_normalization=none"],
            pretrain_name=f"no_layernorm_pretrain_seed{seed}.ckpt",
            tuned_name=f"no_layernorm_tuned_seed{seed}.ckpt",
        )


def task_run_no_layernorm_ablation() -> None:
    root = PROJECT_ROOT / "logs-no-layernorm-ablation"
    settings = ["10+1000@100", "10+10000@100"]
    jobs = [
        {
            "mode": "full",
            "anchor_mode": "full",
            "load_template": "checkpoints/AnchorRevFilter/tuned_seed{seed}.ckpt",
        },
        {
            "mode": "no_layernorm",
            "anchor_mode": "full",
            "extra_algorithm_args": ["algorithm.model.anchor_normalization=none"],
            "load_template": "checkpoints/AnchorRevFilter/no_layernorm_tuned_seed{seed}.ckpt",
        },
    ]
    run_named_anchor_ablation(root, jobs=jobs, settings=settings)
    summarize_anchor_ablation(root, has_seed=True)


def summarize_order_robustness(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*.log")):
        match = re.match(
            r"saif_(original|shuffle\d+)_ckpt(\d+)_(\d+)p(\d+)at(\d+)$",
            path.stem,
        )
        if not match:
            print(f"[WARN] skip {path.stem}")
            continue
        condition, ckpt, n_pos, n_neg, top_k = match.groups()
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "condition": condition,
                "ckpt": ckpt,
                "setting": f"{n_pos}+{n_neg}@{top_k}",
                "density": last_float(r"Avg density:\s*([0-9.]+)", text),
                "HR": last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?)", text),
                "NDCG": last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?)", text),
                "log_path": str(path),
            }
        )

    raw_path = root / "order_robustness_raw.csv"
    md_path = root / "order_robustness_summary.md"
    if DRY_RUN:
        print(f"[dry-run] would write {rel(raw_path)} and {rel(md_path)}")
        return

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "ckpt",
                "setting",
                "density",
                "HR",
                "NDCG",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def mean_std(group: list[dict[str, float | str]], metric: str) -> str:
        values = [r[metric] for r in group if r[metric] is not None]
        if not values:
            return ""
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.6f}+/-{std:.6f}"

    summary_rows = []
    for setting in sorted({r["setting"] for r in rows}):
        conditions = sorted({r["condition"] for r in rows if r["setting"] == setting})
        for condition in conditions:
            group = [
                r
                for r in rows
                if r["setting"] == setting and r["condition"] == condition
            ]
            summary_rows.append(
                {
                    "setting": setting,
                    "condition": condition,
                    "n": len(group),
                    "HR": mean_std(group, "HR"),
                    "NDCG": mean_std(group, "NDCG"),
                }
            )
        shuffled = [
            r
            for r in rows
            if r["setting"] == setting and str(r["condition"]).startswith("shuffle")
        ]
        if shuffled:
            summary_rows.append(
                {
                    "setting": setting,
                    "condition": "shuffled_all",
                    "n": len(shuffled),
                    "HR": mean_std(shuffled, "HR"),
                    "NDCG": mean_std(shuffled, "NDCG"),
                }
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| setting | condition | n | HR | NDCG |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['setting']} | {row['condition']} | {row['n']} | "
                f"{row['HR']} | {row['NDCG']} |\n"
            )
    print(f"Wrote: {rel(raw_path)}")
    print(f"Wrote: {rel(md_path)}")
    print_file(md_path)


def task_run_order_robustness() -> None:
    root = PROJECT_ROOT / "logs-order-robustness"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    settings = ["10+1000@100", "10+10000@100"]
    order_jobs = [("original", "original", 0)] + [
        (f"shuffle{order_seed}", "shuffle", order_seed) for order_seed in [0, 1, 2]
    ]

    for ckpt_id in ["0", "1", "2"]:
        for condition, order, order_seed in order_jobs:
            for setting in settings:
                run_tag = tag(setting)
                cmd = main_cmd(
                    f"SAIFOrder_{condition}_ckpt{ckpt_id}_{run_tag}",
                    [
                        *eval_args(
                            anchor=True,
                            anchor_mode="full",
                            extra_algorithm_args=[
                                f"algorithm.candidate_order={order}",
                                f"algorithm.candidate_order_seed={order_seed}",
                            ],
                        ),
                        f"load=checkpoints/AnchorRevFilter/tuned_seed{ckpt_id}.ckpt",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, root / f"saif_{condition}_ckpt{ckpt_id}_{run_tag}.log")
    summarize_order_robustness(root)


def run_ablation_eval(root: Path, *, modes: Iterable[str], seeds: Iterable[int], include_seed: bool) -> None:
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        for seed in seeds:
            for setting in ["10+1000@100", "10+10000@100"]:
                run_tag = tag(f"{mode}_seed{seed}_{setting}" if include_seed else f"{mode}_{setting}")
                if mode == "full":
                    ckpt_path = f"checkpoints/AnchorRevFilter/tuned_seed{seed}.ckpt"
                else:
                    ckpt_path = f"checkpoints/AnchorRevFilter/{mode}_tuned_seed{seed}.ckpt"
                cmd = main_cmd(
                    f"AnchorAblation_{run_tag}",
                    [
                        *eval_args(anchor=True, anchor_mode=mode),
                        f"load={ckpt_path}",
                        f"+shortcut={setting}",
                    ],
                )
                run_logged(cmd, root / f"{run_tag}.log")


def task_run_ablation_quick() -> None:
    train_anchor_modes(["size_only", "no_balance", "no_density"], [0])
    root = PROJECT_ROOT / "logs-anchor-ablation-quick"
    run_ablation_eval(root, modes=["full", "size_only", "no_balance", "no_density"], seeds=[0], include_seed=False)
    summarize_anchor_ablation(root, has_seed=False)


def task_run_ablation_3seeds() -> None:
    train_anchor_modes(["size_only", "no_balance", "no_density"], [1, 2])
    root = PROJECT_ROOT / "logs-anchor-ablation-3seeds"
    run_ablation_eval(root, modes=["full", "size_only", "no_balance", "no_density"], seeds=[0, 1, 2], include_seed=True)
    summarize_anchor_ablation(root, has_seed=True)


def task_run_symmetric_control() -> None:
    root = PROJECT_ROOT / "logs-symmetric-control"
    if not DRY_RUN:
        root.mkdir(parents=True, exist_ok=True)
    settings = ["10+1000@100", "10+10000@100"]
    for seed in ["0", "1", "2"]:
        for setting in settings:
            run_tag = tag(setting)
            jobs = [
                (
                    "Official",
                    f"official_seed{seed}_{run_tag}.log",
                    eval_args(extra_args=["dataset.eval_pool_mode=symmetric"]),
                    f"checkpoints/RevTrack/{seed}_tuned.ckpt",
                ),
                (
                    "SAIF",
                    f"saif_seed{seed}_{run_tag}.log",
                    eval_args(anchor=True, anchor_mode="full", extra_args=["dataset.eval_pool_mode=symmetric"]),
                    f"checkpoints/AnchorRevFilter/tuned_seed{seed}.ckpt",
                ),
                (
                    "Geometry",
                    f"geometry_seed{seed}_{run_tag}.log",
                    eval_args(anchor=True, anchor_mode="size_only", extra_args=["dataset.eval_pool_mode=symmetric"]),
                    f"checkpoints/SAIFAnchorOnly/tuned_seed{seed}.ckpt",
                ),
            ]
            for label, log_name, args, load in jobs:
                cmd = main_cmd(
                    f"Sym_{label}_seed{seed}_{run_tag}",
                    [*args, f"load={load}", f"+shortcut={setting}"],
                )
                run_logged(cmd, root / log_name)


def task_patch_eval_pool_mode() -> None:
    dataset_path = PROJECT_ROOT / "datasets" / "elliptic" / "dataset.py"
    backup_path = PROJECT_ROOT / "datasets" / "elliptic" / "dataset.py.bak.receiver_balanced"
    config_path = PROJECT_ROOT / "configurations" / "dataset" / "elliptic_recommendation.yaml"

    if DRY_RUN:
        print(f"[dry-run] would patch {rel(dataset_path)} and {rel(config_path)}")
        return

    if not backup_path.exists():
        shutil.copy2(dataset_path, backup_path)
        print(f"Created backup {rel(backup_path)}")

    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    changed = False
    if not any("self.eval_pool_mode" in line for line in lines):
        for i, line in enumerate(lines):
            if "self.num_licits = cfg.num_licits" in line:
                lines.insert(i + 1, '        self.eval_pool_mode = getattr(cfg, "eval_pool_mode", "official")')
                changed = True
                break

    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Official evaluation is receiver-anchored"):
            start = i
            break
    for i, line in enumerate(lines):
        if start is not None:
            break
        if line.strip().startswith("# take both senders and receivers from illicit transactions"):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if line.strip() == "for data in chosen_illicit:":
                start = i
                if i >= 1 and lines[i - 1].strip() == "licit_receiver_candidates = set()":
                    start = i - 1
                while start >= 1 and lines[start - 1].strip().startswith("#"):
                    start -= 1
                break
    if start is not None:
        for i in range(start, len(lines)):
            if lines[i].strip().startswith("senders = torch.tensor"):
                end = i
                break

    if start is None or end is None:
        raise SystemExit("Could not locate candidate-pool construction block.")

    new_block = [
        "        # Official evaluation is receiver-anchored: licit samples add only senders.",
        "        # Control modes add licit receivers to test sensitivity to receiver-pool construction.",
        "        licit_receiver_candidates = set()",
        "        for data in chosen_illicit:",
        "            senders.update(set(data.senders.tolist()))",
        "            receivers.update(set(data.receivers.tolist()))",
        "        for data in chosen_licit:",
        "            senders.update(set(data.senders.tolist()))",
        '            if self.eval_pool_mode == "symmetric":',
        "                receivers.update(set(data.receivers.tolist()))",
        '            elif self.eval_pool_mode == "balanced_receivers":',
        "                licit_receiver_candidates.update(set(data.receivers.tolist()))",
        '            elif self.eval_pool_mode == "official":',
        "                pass",
        "            else:",
        '                raise ValueError(f"Unknown eval_pool_mode={self.eval_pool_mode}")',
        "",
        '        if self.eval_pool_mode == "balanced_receivers":',
        "            num_extra_receivers = min(len(receivers), len(licit_receiver_candidates))",
        "            if num_extra_receivers > 0:",
        "                receivers.update(random.sample(list(licit_receiver_candidates), num_extra_receivers))",
        "",
    ]
    if lines[start:end] != new_block:
        lines = lines[:start] + new_block + lines[end:]
        changed = True

    if changed:
        dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Patched {rel(dataset_path)}")
    else:
        print(f"No changes needed in {rel(dataset_path)}")

    cfg_text = config_path.read_text(encoding="utf-8")
    if "eval_pool_mode:" not in cfg_text:
        config_path.write_text(cfg_text.rstrip() + "\n\neval_pool_mode: official\n", encoding="utf-8")
        print(f"Patched {rel(config_path)}")
    else:
        print("Config already has eval_pool_mode")

    run(python_cmd("-m", "py_compile", str(dataset_path)))


def task_dummy() -> None:
    print("hello world")


Task = tuple[Callable[[], None], str]

TASKS: dict[str, Task] = {
    "anchor-only": (task_anchor_only, "Train SAIF anchor-only pretrain/tuned checkpoints for seeds 0-2."),
    "check-checkpoints": (task_check_checkpoints, "Inspect selected checkpoint tensor shapes."),
    "draw-case-study-43": (task_draw_case_study_43, "Draw the fixed case-43 hit-rank figure."),
    "draw-sparsity-ndcg": (task_draw_sparsity_ndcg, "Draw the sparsity-vs-NDCG curve."),
    "draw-sparsity-hr": (task_draw_sparsity_hr, "Draw the sparsity-vs-HR curve."),
    "draw-topk-hr": (task_draw_topk_hr, "Draw the top-k-vs-HR curve."),
    "draw-topk-ndcg": (task_draw_topk_ndcg, "Draw the top-k-vs-NDCG curve."),
    "evaluate-anchor-only": (task_evaluate_anchor_only, "Evaluate SAIF anchor-only checkpoints on Table 2 settings."),
    "summarize-cost": (task_summarize_cost, "Summarize logs-cost/*.log into cost_summary.md."),
    "summarize-balanced-receiver-control": (
        task_summarize_balanced_receiver_control,
        "Summarize balanced receiver control logs.",
    ),
    "keep-multiplier": (task_keep_multiplier, "Run keep-multiplier sensitivity experiments and summarize deltas."),
    "receiver-balance-control": (task_receiver_balance_control, "Run balanced receiver geometry-control evaluations."),
    "receiver-balanced-instance-wilcoxon": (
        task_receiver_balanced_instance_wilcoxon,
        "Export balanced-receiver instance metrics and run paired Wilcoxon tests.",
    ),
    "run-finetune-ablation": (
        task_run_finetune_ablation,
        "Evaluate SAIF tuned checkpoints against pretrain-only checkpoints.",
    ),
    "run-targeted-ablation": (
        task_run_targeted_ablation,
        "Evaluate full, no-finetune, size-only, no-density, and no-balance SAIF ablations.",
    ),
    "run-order-robustness": (
        task_run_order_robustness,
        "Evaluate SAIF robustness under shuffled sender/receiver candidate order.",
    ),
    "train-no-layernorm": (
        task_train_no_layernorm,
        "Train SAIF checkpoints without anchor LayerNorm for seeds 0-2.",
    ),
    "run-no-layernorm-ablation": (
        task_run_no_layernorm_ablation,
        "Evaluate full SAIF against no-anchor-LayerNorm checkpoints.",
    ),
    "run-3models-topk": (task_run_3models_topk, "Evaluate MLP/NGCF/LightGCN on top-k settings."),
    "run-3models-sparsity": (task_run_3models_sparsity, "Evaluate MLP/NGCF/LightGCN on sparsity settings."),
    "run-baselines-table2": (task_run_baselines_table2, "Evaluate MLP/NGCF/LightGCN on Table 2 settings."),
    "run-anchor": (task_run_anchor, "Evaluate AnchorRevFilter seed0 on the four main settings."),
    "train-anchor-seeds-3-4": (task_train_anchor_seeds_3_4, "Train AnchorRevFilter seeds 3 and 4."),
    "run-anchor-cost": (task_run_anchor_cost, "Run timed AnchorRevFilter cost evaluations."),
    "run-complexity-profile": (
        task_run_complexity_profile,
        "Run three-checkpoint complexity profiling for official RevFilter and SAIF.",
    ),
    "run-anchor-result": (task_run_anchor_result, "Evaluate AnchorRevFilter seeds 0-2 on four main settings."),
    "run-anchor-sparsity": (task_run_anchor_sparsity, "Evaluate AnchorRevFilter on sparsity settings."),
    "run-anchor-topk": (task_run_anchor_topk, "Evaluate AnchorRevFilter on top-k settings."),
    "run-case-study": (task_run_case_study, "Run the detailed RevFilter vs AnchorRevFilter case study."),
    "paired-official-anchor": (task_paired_official_anchor, "Run paired tests for official vs anchor Table 2 raw logs."),
    "compare-anchor-official-table2": (task_compare_anchor_official_table2, "Write Table 2 anchor-vs-official delta markdown."),
    "compare-anchor-official-sparsity": (
        task_compare_anchor_official_sparsity,
        "Write sparsity anchor-vs-official delta markdown.",
    ),
    "build-edge-index": (task_build_edge_index, "Build data/elliptic/processed/edge_index.pt through the dataset loader."),
    "run-official-result": (task_run_official_result, "Evaluate official RevFilter on the four main settings."),
    "run-official-cost": (task_run_official_cost, "Run timed official RevFilter cost evaluations."),
    "summarize-complexity-profile": (
        task_summarize_complexity_profile,
        "Summarize logs-complexity-profile/*.log into mean+/-std complexity tables.",
    ),
    "run-official-sparsity": (task_run_official_sparsity, "Evaluate official RevFilter on sparsity settings."),
    "run-official-topk": (task_run_official_topk, "Evaluate official RevFilter on top-k settings."),
    "train-anchor-seed1": (task_train_anchor_seed1, "Train AnchorRevFilter seed 1."),
    "train-anchor-seed2": (task_train_anchor_seed2, "Train AnchorRevFilter seed 2."),
    "compare-anchor-official-topk": (task_compare_anchor_official_topk, "Write top-k anchor-vs-official delta markdown."),
    "run-anchor-table2": (task_run_anchor_table2, "Evaluate AnchorRevFilter seeds 0-2 on Table 2 settings."),
    "run-official-table2": (task_run_official_table2, "Evaluate official RevFilter seeds 0-2 on Table 2 settings."),
    "run-ablation-quick": (task_run_ablation_quick, "Train seed0 ablations and evaluate quick ablation settings."),
    "run-ablation-3seeds": (task_run_ablation_3seeds, "Train missing ablation seeds and summarize three-seed ablations."),
    "run-symmetric-control": (task_run_symmetric_control, "Run symmetric evaluation-pool control experiments."),
    "patch-eval-pool-mode": (task_patch_eval_pool_mode, "Patch dataset/config to support eval_pool_mode controls."),
    "dummy": (task_dummy, "Compatibility task for scripts/dummy_script.sh."),
}

ALIASES = {
    "run_3model-sparsity": "run-3models-sparsity",
    "run_3models_topk": "run-3models-topk",
    "run_topk-delta": "compare-anchor-official-topk",
    "draw_topk-ncdg": "draw-topk-ndcg",
    "draw_sparsity-h": "draw-sparsity-hr",
}


def list_tasks() -> None:
    width = max(len(name) for name in TASKS)
    for name, (_func, description) in sorted(TASKS.items()):
        print(f"{name:<{width}}  {description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="list", help="Task name, or 'list'.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    global DRY_RUN
    args = parse_args()
    DRY_RUN = args.dry_run
    os.chdir(PROJECT_ROOT)

    task = args.task[:-3] if args.task.endswith(".sh") else args.task
    task = ALIASES.get(task, task)
    if task in {"list", "help", "--help"}:
        list_tasks()
        return
    if task not in TASKS:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        print("Use: python scripts/run_batches.py list", file=sys.stderr)
        raise SystemExit(2)
    TASKS[task][0]()


if __name__ == "__main__":
    main()
