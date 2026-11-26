"""Fit and compose NIR instrument conversions from same-strip paired readings.

A conversion edge maps the six absorbance channels of one instrument onto another.
Edges are fitted by ordinary least squares over same-strip pairs measured on both
instruments.  A relay chain is a path of edges from a replacement instrument back to
the anchor instrument the frozen flexural model was estimated on.  Only the six
absorbance channels are converted: density and thickness are physical strip
properties and are carried through unchanged.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .records import CODE, parse_code, parse_measurement
from .settings import FEATURES, FIELDS, PAIR_COLUMNS, TARGET

PAIR_SPLITS = {"train", "holdout", "pending"}
NIR_FEATURES = tuple(field for field in FEATURES if field.startswith("nir_"))
NIR_COUNT = len(NIR_FEATURES)
PAIR_FIELDS = {**FIELDS, "instrument": "近红外仪器编号"}
EPS = 1e-18


def parse_instrument(value, line: int) -> str:
    if isinstance(value, str) and CODE.fullmatch(value.strip()):
        return value.strip()
    raise ValueError(f"第 {line} 条{PAIR_FIELDS['instrument']}须为 1 至 64 位编号，以英文字母或数字开头。")


def normalize_pairs(rows: list[dict]) -> list[dict]:
    """Validate a paired-reading ledger and keep one record per strip and instrument.

    A physical strip may appear once per instrument but must keep one batch and one
    sample purpose across instruments.  A whole production batch likewise keeps one
    purpose, so a batch is never split between conversion fitting and acceptance.
    """

    seen = set()
    strip_meta: dict[str, tuple[str, str]] = {}
    batch_splits: dict[str, str] = {}
    records = []
    for line, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict) or raw.keys() != set(PAIR_COLUMNS):
            raise ValueError(f"第 {line} 条的列须与仪器对照模板一致：{', '.join(PAIR_COLUMNS)}。")
        row_id = parse_code(raw["id"], "id", line)
        batch_id = parse_code(raw["batch_id"], "batch_id", line)
        instrument = parse_instrument(raw["instrument"], line)
        if (row_id, instrument) in seen:
            raise ValueError(f"第 {line} 条试条 {row_id} 在仪器 {instrument} 上有重复读数。")
        seen.add((row_id, instrument))
        split = raw["split"].strip() if isinstance(raw["split"], str) else ""
        if split not in PAIR_SPLITS:
            raise ValueError(f"第 {line} 条的样本用途须为 train、holdout 或 pending。")
        meta = (batch_id, split)
        if strip_meta.setdefault(row_id, meta) != meta:
            raise ValueError(f"试条 {row_id} 跨制板批或样本用途，请保持同一条试条在各仪器上的划分一致。")
        if batch_splits.setdefault(batch_id, split) != split:
            raise ValueError(f"制板批 {batch_id} 跨样本用途，请将同批试条整批划分。")
        measurements = {field: parse_measurement(raw[field], field, line) for field in FEATURES}
        target = raw[TARGET]
        if split == "pending":
            if target is not None and not (isinstance(target, str) and not target.strip()):
                raise ValueError(f"第 {line} 条待测试条的实测抗折强度须留空。")
            target = None
        else:
            target = parse_measurement(target, TARGET, line)
        records.append({
            "id": row_id, "batch_id": batch_id, "split": split,
            "instrument": instrument, **measurements, TARGET: target,
        })
    return sorted(records, key=lambda row: (row["batch_id"], row["id"], row["instrument"]))


def strip_index(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Index records by strip id and then by instrument."""

    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        index[record["id"]][record["instrument"]] = record
    return index


@dataclass(frozen=True)
class FrozenModel:
    """The prediction part of a completed calibration run, in original units."""

    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    coefficient: np.ndarray
    target_mean: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.target_mean + (x - self.mean) @ self.coefficient

    def outside(self, x: np.ndarray) -> np.ndarray:
        return (x < self.low) | (x > self.high)


def frozen_model_from_result(result: dict | None) -> FrozenModel:
    if not isinstance(result, dict):
        raise ValueError("原模型运行尚未完成，无法读取冻结校准。")
    try:
        target_mean = float(result["metrics"]["训练抗折均值（MPa）"])
        table = next(item for item in result["tables"] if item.get("key") == "frozen_calibration")
        rows = {row["feature"]: row for row in table["rows"]}
        mean = np.array([float(rows[field]["train_mean"]) for field in FEATURES])
        low = np.array([float(rows[field]["train_min"]) for field in FEATURES])
        high = np.array([float(rows[field]["train_max"]) for field in FEATURES])
        coefficient = np.array([float(rows[field]["coefficient_mpa"]) for field in FEATURES])
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise ValueError("原模型结果缺少冻结校准表或训练抗折均值，无法建立换算。") from error
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(low)) and np.all(np.isfinite(high))
            and np.all(np.isfinite(coefficient)) and np.isfinite(target_mean)):
        raise ValueError("原模型冻结校准包含非有限数值，无法建立换算。")
    return FrozenModel(mean, low, high, coefficient, target_mean)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    intercept: np.ndarray
    slope: np.ndarray
    pairs: int
    fit_batches: tuple[str, ...]

    @property
    def batches(self) -> int:
        return len(self.fit_batches)


@dataclass(frozen=True)
class Chain:
    """A relay path from one instrument back to the anchor instrument."""

    chain_id: int
    path: tuple[str, ...]
    intercept: np.ndarray
    slope: np.ndarray

    @property
    def source(self) -> str:
        return self.path[0]

    @property
    def anchor(self) -> str:
        return self.path[-1]

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def label(self) -> str:
        return "→".join(self.path)

    def apply(self, absorbance: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * absorbance


def _affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mean_source = float(source.mean())
    mean_target = float(target.mean())
    centered = source - mean_source
    sxx = float(centered @ centered)
    if sxx <= EPS:
        return mean_target - mean_source, 1.0
    slope = float(centered @ (target - mean_target) / sxx)
    return mean_target - slope * mean_source, slope


def _fit_direction(points: list[tuple[dict, dict]], source: str, target: str) -> Edge:
    source_x = np.array([[record[field] for field in NIR_FEATURES] for record, _ in points], dtype=float)
    target_x = np.array([[record[field] for field in NIR_FEATURES] for _, record in points], dtype=float)
    intercept = np.zeros(NIR_COUNT)
    slope = np.ones(NIR_COUNT)
    for channel in range(NIR_COUNT):
        intercept[channel], slope[channel] = _affine(source_x[:, channel], target_x[:, channel])
        if not (np.isfinite(intercept[channel]) and np.isfinite(slope[channel])):
            raise ValueError(f"仪器 {source} 到 {target} 的 {NIR_FEATURES[channel]} 换算不可拟合。")
    return Edge(source, target, intercept, slope, len(points),
                tuple(sorted({record["batch_id"] for record, _ in points})))


def fit_edges(records: list[dict]) -> dict[tuple[str, str], Edge]:
    """Fit both directions of every instrument pair from training-split strips."""

    pairs: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for instruments in strip_index(records).values():
        for first, second in combinations(sorted(instruments), 2):
            left, right = instruments[first], instruments[second]
            if left["split"] == "train" and right["split"] == "train":
                pairs[(first, second)].append((left, right))
    edges = {}
    for (first, second), points in sorted(pairs.items()):
        # Two same-strip pairs are the minimum for a least-squares slope.
        if len(points) < 2:
            continue
        edges[(first, second)] = _fit_direction(points, first, second)
        edges[(second, first)] = _fit_direction([(right, left) for left, right in points], second, first)
    return edges


def enumerate_chains(edges: dict[tuple[str, str], Edge], source: str, anchor: str,
                     maximum_hops: int) -> list[Chain]:
    """List simple relay chains from a replacement instrument back to the anchor."""

    if source == anchor:
        return [Chain(0, (anchor,), np.zeros(NIR_COUNT), np.ones(NIR_COUNT))]
    adjacency: dict[str, list[str]] = defaultdict(list)
    for start, end in edges:
        adjacency[start].append(end)
    for destinations in adjacency.values():
        destinations.sort()
    paths: list[tuple[str, ...]] = []

    def walk(path: list[str]):
        if path[-1] == anchor:
            paths.append(tuple(path))
            return
        if len(path) - 1 >= maximum_hops:
            return
        for destination in adjacency.get(path[-1], ()):
            if destination not in path:
                walk([*path, destination])

    walk([source])
    if len(paths) > 200:
        raise ValueError("可用仪器接力链超过 200 条，请降低接力跳数上限或精简对照仪器。")
    chains = []
    for chain_id, path in enumerate(sorted(paths, key=lambda item: (len(item), item)), start=1):
        intercept = np.zeros(NIR_COUNT)
        slope = np.ones(NIR_COUNT)
        for start, end in zip(path, path[1:]):
            edge = edges[(start, end)]
            intercept = edge.intercept + edge.slope * intercept
            slope = edge.slope * slope
        chains.append(Chain(chain_id, path, intercept, slope))
    return chains


def absorbance(record: dict) -> np.ndarray:
    return np.array([record[field] for field in NIR_FEATURES], dtype=float)


def feature_vector(record: dict, nir: np.ndarray | None = None) -> np.ndarray:
    """Assemble the model input, replacing only the absorbance channels."""

    converted = absorbance(record) if nir is None else np.asarray(nir, dtype=float)
    values = []
    channel = 0
    for field in FEATURES:
        if field in NIR_FEATURES:
            values.append(float(converted[channel]))
            channel += 1
        else:
            values.append(float(record[field]))
    return np.array(values, dtype=float)
