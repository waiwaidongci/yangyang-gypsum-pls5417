"""Validate relay chains and prepare an instrument-swap handoff.

The frozen flexural model was estimated on the anchor instrument.  Readings from a
replacement instrument are mapped back through one or more fitted conversion edges
before the model is applied.  A chain is usable only when whole holdout batches
measured on both endpoints of the chain reproduce the anchor estimate within the
agreed error line; those acceptance batches never enter that chain's edge fitting.

Pending strips are handed off only when every strip has at least one qualified chain
and every qualified chain reaches the same decision.  Otherwise the handoff is
deferred.  An issued handoff is an immutable snapshot of its decision and basis.
When late comparison data is folded into a re-review, the formal handoff keeps only
strips that still have a qualified chain and whose every qualified chain supports the
original decision; the whole table switches at once.
"""

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .instruments import (
    NIR_COUNT, NIR_FEATURES, Chain, Edge, FrozenModel, absorbance, enumerate_chains,
    feature_vector, fit_edges, frozen_model_from_result, normalize_pairs, strip_index,
)
from .settings import FEATURES

DEFAULT_MAX_HOPS = 2
DEFAULT_ERROR_MPA = 0.30
MAX_HOPS_LIMIT = 4
MIN_ACCEPTANCE_STRIPS = 2

PRIORITY_ACTIONS = {
    1: "核对读数并优先实测",
    2: "低于筛查线，优先实测",
    3: "按班次常规实测",
}

HANDOFF_LABELS = {
    "position": "交接顺序", "instrument": "近红外仪器", "id": "试条编号", "batch_id": "制板批号",
    "representative_chain": "代表合格链", "predicted_mpa": "估计抗折（MPa）",
    "outside_features": "超出训练范围的字段", "screening_side": "相对筛查线的判定",
    "priority": "实测优先级（1最高）", "action": "实测安排",
    "status": "交接状态", "qualified_chains": "合格链数", "reason": "暂缓或争议说明",
}


def parse_conversion_settings(settings: dict) -> tuple[int, float]:
    if not isinstance(settings, dict) or settings.keys() - {"max_hops", "error_mpa"}:
        raise ValueError("换算参数仅接受 max_hops 和 error_mpa。")
    hops = settings.get("max_hops", DEFAULT_MAX_HOPS)
    limit = settings.get("error_mpa", DEFAULT_ERROR_MPA)
    if type(hops) not in (int, float) or hops != int(hops) or not 1 <= hops <= MAX_HOPS_LIMIT:
        raise ValueError(f"接力跳数上限须为 1 至 {MAX_HOPS_LIMIT} 的整数。")
    if type(limit) not in (int, float) or not 0.01 <= limit <= 5:
        raise ValueError("约定误差线须为 0.01 至 5 的数值（MPa）。")
    return int(hops), float(limit)


def merge_pair_rows(groups: list[list[dict]]) -> list[dict]:
    """Merge paired ledgers, rejecting the same strip/instrument read differently."""

    merged: dict[tuple[str, str], dict] = {}
    for rows in groups:
        for record in normalize_pairs(rows):
            key = (record["id"], record["instrument"])
            if key in merged and merged[key] != record:
                raise ValueError(f"对照数据中试条 {record['id']} 在仪器 {record['instrument']} 上存在不一致的重复读数。")
            merged[key] = record
    return sorted(merged.values(), key=lambda row: (row["batch_id"], row["id"], row["instrument"]))


def _judge(model: FrozenModel, record: dict, chain: Chain, screening: float) -> dict:
    converted = chain.apply(absorbance(record))
    features = feature_vector(record, converted)
    prediction = float(model.predict(features.reshape(1, -1))[0])
    outside = [field for field, flag in zip(FEATURES, model.outside(features.reshape(1, -1))[0]) if flag]
    if outside:
        priority = 1
    elif prediction < screening:
        priority = 2
    else:
        priority = 3
    return {
        "predicted_mpa": round(prediction, 8), "priority": priority,
        "action": PRIORITY_ACTIONS[priority], "outside_features": ",".join(outside),
        "screening_side": "低于筛查线" if prediction < screening else "不低于筛查线",
    }


def _acceptance(model: FrozenModel, records: list[dict], chain: Chain) -> dict:
    """Score a chain on whole holdout batches measured on both of its endpoints."""

    strips = strip_index(records)
    by_batch: dict[str, list[str]] = defaultdict(list)
    for strip_id, instruments in strips.items():
        any_record = next(iter(instruments.values()))
        if any_record["split"] == "holdout":
            by_batch[any_record["batch_id"]].append(strip_id)
    used, excluded = [], []
    for batch_id in sorted(by_batch):
        ids = sorted(by_batch[batch_id])
        complete = all(chain.source in strips[strip_id] and chain.anchor in strips[strip_id] for strip_id in ids)
        if complete:
            used.append((batch_id, ids))
        else:
            excluded.append(batch_id)
    pairs = []
    for batch_id, ids in used:
        for strip_id in ids:
            source = strips[strip_id][chain.source]
            anchor = strips[strip_id][chain.anchor]
            reference = float(model.predict(feature_vector(anchor).reshape(1, -1))[0])
            converted = float(model.predict(feature_vector(source, chain.apply(absorbance(source))).reshape(1, -1))[0])
            pairs.append({
                "id": strip_id, "batch_id": batch_id,
                "reference_mpa": reference, "converted_mpa": converted, "error_mpa": converted - reference,
            })
    errors = np.array([pair["error_mpa"] for pair in pairs]) if pairs else np.array([])
    return {
        "pairs": pairs, "batches": [batch_id for batch_id, _ in used], "excluded_batches": excluded,
        "rmse_mpa": float(np.sqrt(np.mean(errors ** 2))) if pairs else None,
        "bias_mpa": float(np.mean(errors)) if pairs else None,
        "max_abs_mpa": float(np.max(np.abs(errors))) if pairs else None,
    }


def _qualified(acceptance: dict, limit: float) -> tuple[bool, str]:
    if not acceptance["batches"]:
        return False, "缺少未参与拟合的整批首尾双机验收试条"
    if len(acceptance["pairs"]) < MIN_ACCEPTANCE_STRIPS:
        return False, f"整批验收试条不足 {MIN_ACCEPTANCE_STRIPS} 条"
    if acceptance["rmse_mpa"] > limit:
        return False, f"验收 RMSE 超过约定误差线 {limit:g} MPa"
    return True, "验收通过"


def _acceptance_rows(chain: Chain, acceptance: dict, qualified: bool, limit: float) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in acceptance["pairs"]:
        grouped[pair["batch_id"]].append(pair["error_mpa"])
    rows = []
    for scope, batch, errors in [
        ("overall", "", [pair["error_mpa"] for pair in acceptance["pairs"]]),
        *(("batch", batch, grouped[batch]) for batch in sorted(grouped)),
    ]:
        vector = np.array(errors) if errors else np.array([])
        rows.append({
            "scope": scope, "chain_id": chain.chain_id, "path": chain.label, "batch_id": batch,
            "strips": len(errors),
            "rmse_mpa": round(float(np.sqrt(np.mean(vector ** 2))), 8) if len(vector) else None,
            "mae_mpa": round(float(np.mean(np.abs(vector))), 8) if len(vector) else None,
            "bias_mpa": round(float(np.mean(vector)), 8) if len(vector) else None,
            "error_limit_mpa": limit, "qualified": qualified if scope == "overall" else None,
        })
    return rows


def _evaluate_chains(model: FrozenModel, records: list[dict], edges: dict[tuple[str, str], Edge],
                     anchor: str, maximum: int, limit: float) -> list[dict]:
    instruments = sorted({record["instrument"] for record in records})
    evaluated = [{
        "chain": Chain(0, (anchor,), np.zeros(NIR_COUNT), np.ones(NIR_COUNT)),
        "qualified": True, "rmse": 0.0, "bias": 0.0, "max_abs": 0.0, "note": "锚定仪器直测",
        "acceptance_batches": 0, "acceptance_strips": 0, "acceptance_rows": [],
    }]
    next_id = 0
    for source in instruments:
        if source == anchor:
            continue
        for chain in enumerate_chains(edges, source, anchor, maximum):
            next_id += 1
            chain = replace(chain, chain_id=next_id)
            acceptance = _acceptance(model, records, chain)
            qualified, note = _qualified(acceptance, limit)
            evaluated.append({
                "chain": chain, "qualified": qualified,
                "rmse": acceptance["rmse_mpa"], "bias": acceptance["bias_mpa"],
                "max_abs": acceptance["max_abs_mpa"], "note": note,
                "acceptance_batches": len(acceptance["batches"]),
                "acceptance_strips": len(acceptance["pairs"]),
                "acceptance_rows": _acceptance_rows(chain, acceptance, qualified, limit),
            })
    return evaluated


def _pending_analysis(model: FrozenModel, records: list[dict], evaluated: list[dict], screening: float):
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in evaluated:
        by_source[item["chain"].source].append(item)
    for items in by_source.values():
        items.sort(key=lambda item: (item["chain"].hops, item["chain"].chain_id))
    per_chain_rows, strip_rows = [], []
    for record in records:
        if record["split"] != "pending":
            continue
        candidates = by_source.get(record["instrument"], [])
        usable = [item for item in candidates if item["qualified"]]
        judgments = []
        for item in usable:
            judged = _judge(model, record, item["chain"], screening)
            judgments.append((item, judged))
            if item["chain"].hops > 0:
                per_chain_rows.append({
                    "instrument": record["instrument"], "id": record["id"], "batch_id": record["batch_id"],
                    "chain_id": item["chain"].chain_id, "path": item["chain"].label,
                    "predicted_mpa": judged["predicted_mpa"], "priority": judged["priority"],
                    "action": judged["action"], "outside_features": judged["outside_features"],
                })
        if not judgments:
            strip_rows.append({
                "instrument": record["instrument"], "id": record["id"], "batch_id": record["batch_id"],
                "predicted_mpa": None, "outside_features": "", "priority": None,
                "action": "暂缓交接，补测首尾双机对照", "status": "unconverted",
                "qualified_chains": 0, "representative_chain": "", "reason": "该仪器没有合格接力链",
                "screening_side": "", "chain_sides": [],
            })
            continue
        # Prefer the tightest qualified chain: lowest acceptance RMSE, then fewest hops.
        representative = min(usable, key=lambda item: (item["rmse"], item["chain"].hops, item["chain"].chain_id))
        chosen = next(judged for item, judged in judgments if item is representative)
        # The judgement that must agree across chains is which side of the screening
        # line the strip falls on; otherwise the handoff is deferred.
        sides = {judged["screening_side"] for _, judged in judgments}
        settled = len(sides) == 1
        strip_rows.append({
            "instrument": record["instrument"], "id": record["id"], "batch_id": record["batch_id"],
            "predicted_mpa": chosen["predicted_mpa"],
            "outside_features": chosen["outside_features"], "priority": chosen["priority"],
            "action": chosen["action"], "status": "settled" if settled else "disputed",
            "qualified_chains": len(judgments), "representative_chain": representative["chain"].label,
            "reason": "" if settled else "合格链判在筛查线两侧",
            "screening_side": chosen["screening_side"],
            "chain_sides": [judged["screening_side"] for _, judged in judgments],
        })
    strip_rows.sort(key=lambda row: (
        row["priority"] if row["priority"] is not None else 99,
        row["predicted_mpa"] if row["predicted_mpa"] is not None else float("inf"),
        row["id"],
    ))
    for position, row in enumerate(strip_rows, start=1):
        row["position"] = position
    return per_chain_rows, strip_rows


def _revision(original_rows: list[dict], current: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only strips whose every qualified chain supports the original verdict.

    The verdict is the screening-line side.  A strip is retained only if it still has
    a qualified chain and *every* qualified chain agrees with the side recorded in the
    original handoff.  Everything else is listed as dropped; the formal table switches
    as a whole.
    """

    originals = {(row["instrument"], row["id"]): row for row in original_rows}
    current_by_key = {(row["instrument"], row["id"]): row for row in current}
    retained, dropped = [], []
    for key, before in originals.items():
        now = current_by_key.get(key)
        if now is None:
            dropped.append({
                "id": before["id"], "instrument": before["instrument"],
                "original_priority": before["priority"], "status": "no_current_reading",
                "reason": "本次对照台账没有该试条读数",
            })
            continue
        if now["status"] == "unconverted":
            dropped.append({
                "id": now["id"], "instrument": now["instrument"],
                "original_priority": before["priority"], "status": "unconverted",
                "reason": "本次不再有合格链",
            })
        elif now["status"] == "disputed" or any(
                side != before["screening_side"] for side in now["chain_sides"]):
            dropped.append({
                "id": now["id"], "instrument": now["instrument"],
                "original_priority": before["priority"], "status": "judgement_changed",
                "reason": "合格链不再支持原判定" if now["status"] != "disputed"
                else "合格链判在筛查线两侧，不再支持唯一原判定",
            })
        else:
            retained.append({**now, "original_priority": before["priority"]})
    # New pending strips are outside the original handoff and are not auto-added.
    for key, now in current_by_key.items():
        if key not in originals:
            dropped.append({
                "id": now["id"], "instrument": now["instrument"],
                "original_priority": None, "status": "not_in_original",
                "reason": "不在原交接范围内，本次不扩展下载表",
            })
    retained.sort(key=lambda row: (row["priority"], row["predicted_mpa"], row["id"]))
    for position, row in enumerate(retained, start=1):
        row["position"] = position
    return retained, dropped


def _table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def _edge_rows(edges: dict[tuple[str, str], Edge]) -> list[dict]:
    rows = []
    for (source, target), edge in sorted(edges.items()):
        for channel, field in enumerate(NIR_FEATURES):
            rows.append({
                "source": source, "target": target, "feature": field,
                "intercept": round(float(edge.intercept[channel]), 8),
                "slope": round(float(edge.slope[channel]), 8),
                "pairs": edge.pairs, "batches": edge.batches,
                "fit_batches": ",".join(edge.fit_batches),
            })
    return rows


def _chain_row(item: dict, limit: float) -> dict:
    chain = item["chain"]
    return {
        "chain_id": chain.chain_id, "path": chain.label, "hops": chain.hops,
        "acceptance_batches": item["acceptance_batches"], "acceptance_strips": item["acceptance_strips"],
        "acceptance_rmse_mpa": round(item["rmse"], 8) if item["rmse"] is not None else None,
        "acceptance_bias_mpa": round(item["bias"], 8) if item["bias"] is not None else None,
        "acceptance_max_abs_mpa": round(item["max_abs"], 8) if item["max_abs"] is not None else None,
        "error_limit_mpa": limit, "qualified": item["qualified"], "note": item["note"],
    }


def analyze_conversion(rows: list[dict], base_result: dict, anchor: str, settings: dict,
                       screening: float, previous_handoff: dict | None) -> dict:
    records = normalize_pairs(rows)
    if not records:
        raise ValueError("仪器对照台账至少需要一条读数。")
    model = frozen_model_from_result(base_result)
    maximum, limit = parse_conversion_settings(settings)
    instruments = sorted({record["instrument"] for record in records})
    if anchor not in instruments:
        raise ValueError(f"原模型锚定仪器 {anchor} 在对照台账中没有读数。")
    edges = fit_edges(records)
    evaluated = _evaluate_chains(model, records, edges, anchor, maximum, limit)
    chain_items = [item for item in evaluated if item["chain"].hops > 0]
    qualified = sum(1 for item in chain_items if item["qualified"])
    per_chain_rows, strips = _pending_analysis(model, records, evaluated, screening)

    if previous_handoff is not None:
        formal, dropped = _revision(previous_handoff["rows"], strips)
        reasons = []
        if not formal:
            reasons.append("复核后没有仍被合格链支持原判定的试条，整表暂缓切换。")
        ready = bool(formal)
        revision = True
    else:
        reasons = []
        if not strips:
            reasons.append("本次对照台账没有待测试条。")
        if any(row["status"] == "unconverted" for row in strips):
            reasons.append("存在没有合格链的待测试条。")
        if any(row["status"] == "disputed" for row in strips):
            reasons.append("合格链把部分试条判在筛查线两侧或判定不一致。")
        # A single strip without a qualified chain, or with chains split across the
        # screening line, defers the whole handoff: no formal table is issued.
        ready = bool(strips) and not reasons
        formal = [row for row in strips if row["status"] == "settled"] if ready else []
        dropped = []
        revision = False

    display = [{key: row.get(key) for key in HANDOFF_LABELS} for row in formal]
    handoff = {
        "anchor_instrument": anchor, "error_mpa": limit, "max_hops": maximum,
        "screening_mpa": screening, "ready": ready, "reasons": reasons, "revision": revision,
        "previous_handoff_id": previous_handoff["handoff_id"] if previous_handoff else None,
        "rows": display,
        "dropped": [{
            "id": row["id"], "instrument": row["instrument"], "original_priority": row["original_priority"],
            "status": row["status"], "reason": row["reason"],
        } for row in dropped],
    }
    metrics = {
        "锚定仪器": anchor, "对照仪器数": len(instruments), "换算边数": len(edges),
        "接力链数": len(chain_items), "合格链数": qualified, "约定误差线（MPa）": limit,
        "待测试条数": len(strips), "正式交接试条数": len(formal), "暂缓交接": not ready,
    }
    summary = (
        f"以 {anchor} 为锚定仪器，从 {len(edges)} 条换算边枚举 {len(chain_items)} 条接力链，"
        f"按 {limit:g} MPa 误差线验收后保留 {qualified} 条合格链。"
    )
    if revision:
        summary += (f"本次为对交接 {previous_handoff['handoff_id']} 的复核，正式表保留 {len(formal)} 条、"
                    f"剔除 {len(dropped)} 条，整表一次性切换。")
    summary += (f"待测 {len(strips)} 条，正式交接 {len(formal)} 条。" if ready
                else "本次交接暂缓，不生成正式表：" + "；".join(reasons))

    tables = [
        _table("instrument_edges", "仪器间换算边（OLS，仅六个吸光度通道）", {
            "source": "源仪器", "target": "目标仪器", "feature": "测量字段", "intercept": "截距",
            "slope": "斜率", "pairs": "拟合试条对数", "batches": "拟合制板批数", "fit_batches": "参与拟合的制板批",
        }, _edge_rows(edges)),
        _table("relay_chains", "接力链验收", {
            "chain_id": "链编号", "path": "接力路径", "hops": "接力跳数",
            "acceptance_batches": "整批验收批数", "acceptance_strips": "整批验收试条数",
            "acceptance_rmse_mpa": "验收 RMSE（MPa）", "acceptance_bias_mpa": "验收平均偏差（MPa）",
            "acceptance_max_abs_mpa": "验收最大绝对误差（MPa）",
            "error_limit_mpa": "约定误差线（MPa）", "qualified": "是否合格", "note": "判定依据",
        }, [_chain_row(item, limit) for item in chain_items]),
        _table("chain_acceptance", "合格链逐批验收误差", {
            "scope": "汇总层级", "chain_id": "链编号", "path": "接力路径", "batch_id": "验收制板批",
            "strips": "试条数", "rmse_mpa": "RMSE（MPa）", "mae_mpa": "MAE（MPa）",
            "bias_mpa": "平均换算偏差（MPa）", "error_limit_mpa": "约定误差线（MPa）", "qualified": "是否合格",
        }, [row for item in chain_items for row in item["acceptance_rows"]]),
        _table("pending_by_chain", "待测试条的各合格链估计", {
            "instrument": "近红外仪器", "id": "试条编号", "batch_id": "制板批号", "chain_id": "链编号",
            "path": "接力路径", "predicted_mpa": "估计抗折（MPa）", "priority": "实测优先级（1最高）",
            "action": "实测安排", "outside_features": "超出训练范围的字段",
        }, per_chain_rows),
        _table("pending_estimates", "待测试条及实测优先顺序", HANDOFF_LABELS, display),
    ]
    if revision:
        tables.append(_table("revision_dropped", "复核后不再正式交接的试条", {
            "id": "试条编号", "instrument": "近红外仪器", "original_priority": "原判定优先级",
            "status": "本次状态", "reason": "剔除原因",
        }, handoff["dropped"]))
    else:
        tables.append(_table("pending_disposition", "待测试条交接状态", {
            "instrument": "近红外仪器", "id": "试条编号", "batch_id": "制板批号", "status": "交接状态",
            "priority": "实测优先级（1最高）", "qualified_chains": "合格链数", "reason": "暂缓或争议说明",
        }, [{
            "instrument": row["instrument"], "id": row["id"], "batch_id": row["batch_id"],
            "status": row["status"], "priority": row["priority"],
            "qualified_chains": row["qualified_chains"], "reason": row["reason"],
        } for row in strips]))
    return {"summary": summary, "metrics": metrics, "tables": tables,
            "warnings": list(reasons), "handoff": handoff}
