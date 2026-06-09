#!/usr/bin/env python
"""Shared helpers for edge-recommendation result analysis scripts."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def parse_setting(setting: str) -> tuple[int, int, int]:
    dataset_part, top_k = setting.split("@", 1)
    num_illicits, num_licits = dataset_part.split("+", 1)
    return int(num_illicits), int(num_licits), int(top_k)


def setting_sort_key(setting: str) -> tuple[int, int, int]:
    if setting in TABLE2_SETTINGS:
        return (0, TABLE2_SETTINGS.index(setting), 0)
    num_illicits, num_licits, top_k = parse_setting(setting)
    return (1, num_illicits, num_licits, top_k)


def tag(text: str) -> str:
    return text.replace("+", "p").replace("@", "at").replace(".", "p")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def mean_std(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return f"{mean(values):.6f} +/- {sample_std(values):.6f}"


def dcg_at_k(k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(k))


def choose_config_name(overrides: Iterable[str], key: str, default: str) -> str:
    prefix = f"{key}="
    for override in reversed(list(overrides)):
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    return default


def compose_edge_cfg(
    *,
    setting: str,
    algorithm: str,
    seed: int,
    overrides: Iterable[str] = (),
):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    from utils.exp_utils import override_exp_edge_recommendation_cfg

    base_overrides = [
        "experiment=exp_edge_recommendation",
        "dataset=elliptic_recommendation",
        f"algorithm={algorithm}",
        "experiment.tasks=[test]",
        "experiment.test.batch_size=16",
        "experiment.test.data.num_workers=0",
        f"seed={seed}",
        "wandb.mode=offline",
        f"+shortcut={setting}",
        *list(overrides),
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configurations"),
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=base_overrides)

    experiment_name = choose_config_name(
        base_overrides, "experiment", "exp_edge_recommendation"
    )
    dataset_name = choose_config_name(
        base_overrides, "dataset", "elliptic_recommendation"
    )
    algorithm_name = choose_config_name(base_overrides, "algorithm", algorithm)

    with open_dict(cfg):
        cfg.experiment._name = experiment_name
        cfg.dataset._name = dataset_name
        cfg.algorithm._name = algorithm_name
        override_exp_edge_recommendation_cfg(cfg)

    OmegaConf.resolve(cfg)
    return cfg
