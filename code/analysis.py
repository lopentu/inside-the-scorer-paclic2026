from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from registry import Registry

DEFAULT_RELIABILITY_THRESHOLD = 0.6
DEFAULT_RESAMPLES = 50_000

PAPER_MODEL_IDS = [
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
    "ibm-granite/granite-4.1-8b-base",
    "nvidia/Mistral-NeMo-Minitron-8B-Base",
    "google/gemma-2-2b",
    "google/gemma-3-1b-pt",
    "google/gemma-3-270m",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "taide/TAIDE-LX-7B",
    "meta-llama/Llama-2-7b-hf",
    "google/gemma-3-4b-pt",
    "allenai/OLMo-2-0425-1B",
    "mistralai/Mistral-7B-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM3-3B-Base",
    "ibm-granite/granite-3.3-2b-base",
    "ibm-granite/granite-4.1-3b-base",
    "ibm-granite/granite-3.3-8b-base",
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-7B-Base",
    "tiiuae/Falcon3-10B-Base",
    "HuggingFaceTB/SmolLM2-360M",
    "swiss-ai/Apertus-8B-2509",
]


def paper_model_ids() -> list[str]:
    return list(PAPER_MODEL_IDS)


OUTPUT_KINDS = ("score", "probe", "geometry", "steer")
RunScope = dict[str, dict[str, str]]


def _selected_run_id(run_scope: RunScope | None, output_kind: str, model_hf_id: str) -> str | None:
    if run_scope is None:
        return None
    return run_scope.get(output_kind, {}).get(model_hf_id)


def _table_query_where(
    model_hf_id: str, output_kind: str, run_scope: RunScope | None,
) -> tuple[str, tuple[str, ...]]:
    run_id = _selected_run_id(run_scope, output_kind, model_hf_id)
    if run_scope is not None:
        if run_id is None:
            return " AND 0", ()
        return " AND r.id = ?", (run_id,)
    return " AND r.model_hf_id = ?", (model_hf_id,)


def _expected_output_rows(output_kind: str, n_layers: int | None) -> int | None:
    if output_kind == "score":
        return 75
    if n_layers is None:
        return 1
    if output_kind == "probe":
        return n_layers * 3
    if output_kind == "geometry":
        return n_layers * 66
    if output_kind == "steer":
        return n_layers * 864
    raise ValueError(f"unknown output kind: {output_kind}")


def _output_run_candidates(registry: "Registry", model_hf_id: str, output_kind: str) -> list[dict]:
    if output_kind == "score":
        return [dict(row) for row in registry.query(
            "SELECT r.id, r.phase, r.status, r.started_at, r.finished_at,"
            " COUNT(*) AS row_count, COUNT(DISTINCT x.candidate_id || ':' || x.part) AS coverage_count"
            " FROM experiment_runs r JOIN model_scores x ON x.run_id = r.id"
            " WHERE r.model_hf_id = ? AND r.finished_at IS NOT NULL"
            " GROUP BY r.id ORDER BY r.finished_at DESC",
            (model_hf_id,))]
    if output_kind == "probe":
        return [dict(row) for row in registry.query(
            "SELECT r.id, r.phase, r.status, r.started_at, r.finished_at,"
            " COUNT(*) AS row_count,"
            " COUNT(DISTINCT CASE WHEN x.form_scope = 'pooled'"
            " THEN x.layer || ':' || x.part END) AS coverage_count"
            " FROM experiment_runs r JOIN probes x ON x.run_id = r.id"
            " WHERE r.model_hf_id = ? AND r.finished_at IS NOT NULL"
            " GROUP BY r.id ORDER BY r.finished_at DESC",
            (model_hf_id,))]
    if output_kind == "geometry":
        return [dict(row) for row in registry.query(
            "SELECT r.id, r.phase, r.status, r.started_at, r.finished_at,"
            " COUNT(*) AS row_count, COUNT(DISTINCT x.layer) AS coverage_count"
            " FROM experiment_runs r JOIN contrastive_geometry x ON x.run_id = r.id"
            " WHERE r.model_hf_id = ? AND r.finished_at IS NOT NULL"
            " GROUP BY r.id ORDER BY r.finished_at DESC",
            (model_hf_id,))]
    if output_kind == "steer":
        return [dict(row) for row in registry.query(
            "SELECT r.id, r.phase, r.status, r.started_at, r.finished_at,"
            " COUNT(*) AS row_count, COUNT(DISTINCT x.layer) AS coverage_count"
            " FROM experiment_runs r JOIN steering_metrics x ON x.run_id = r.id"
            " WHERE r.model_hf_id = ? AND r.finished_at IS NOT NULL"
            " GROUP BY r.id ORDER BY r.finished_at DESC",
            (model_hf_id,))]
    raise ValueError(f"unknown output kind: {output_kind}")


def _is_complete_output(candidate: dict, output_kind: str, expected_rows: int | None,
                        n_layers: int | None) -> bool:
    row_count = int(candidate["row_count"] or 0)
    coverage = int(candidate["coverage_count"] or 0)
    if expected_rows is None:
        return row_count > 0
    if output_kind == "score":
        return row_count >= expected_rows
    if output_kind == "probe":
        return coverage >= expected_rows
    if output_kind in ("geometry", "steer"):
        return row_count >= expected_rows and (n_layers is None or coverage >= n_layers)
    return False


def model_catalog_lookup(registry: "Registry", model_hf_ids: list[str]) -> dict[str, dict]:
    if not model_hf_ids:
        return {}
    placeholders = ",".join("?" for _ in model_hf_ids)
    rows = registry.query(
        f"SELECT hf_id, family, org, country, params_b, n_layers, d_model, base_model,"
        f" catalog, training_status, published_date"
        f" FROM models WHERE hf_id IN ({placeholders})", tuple(model_hf_ids))
    return {row["hf_id"]: dict(row) for row in rows}


def build_run_audit(
    registry: "Registry", model_hf_ids: list[str],
) -> tuple[RunScope, list[dict]]:
    catalog = model_catalog_lookup(registry, model_hf_ids)
    scope: RunScope = {kind: {} for kind in OUTPUT_KINDS}
    audit: list[dict] = []
    for model_hf_id in model_hf_ids:
        n_layers = catalog.get(model_hf_id, {}).get("n_layers")
        for output_kind in OUTPUT_KINDS:
            expected_rows = _expected_output_rows(output_kind, n_layers)
            candidates = _output_run_candidates(registry, model_hf_id, output_kind)
            selected = next(
                (row for row in candidates
                 if _is_complete_output(row, output_kind, expected_rows, n_layers)),
                None)
            if selected:
                scope[output_kind][model_hf_id] = selected["id"]
            latest = candidates[0] if candidates else {}
            audit.append({
                "model_hf_id": model_hf_id,
                "output_kind": output_kind,
                "selected_run_id": selected.get("id") if selected else None,
                "complete": selected is not None,
            })
    return scope, audit


class Phase1PartSummary(TypedDict):
    n_layers: int
    best_r2: float
    peak_layer: int
    peak_perm_pvalue: float | None
    final_r2: float
    relative_drop: float | None


def phase1_layer_summary(
    registry: "Registry", model_hf_id: str, form_scope: str = "pooled",
    run_scope: RunScope | None = None,
) -> dict[int, Phase1PartSummary]:
    where, params = _table_query_where(model_hf_id, "probe", run_scope)
    rows = registry.query(
        "SELECT p.part, p.layer, p.r2_loocv, p.perm_pvalue FROM probes p"
        " JOIN experiment_runs r ON r.id = p.run_id"
        " WHERE p.form_scope = ?" + where,
        (form_scope, *params))
    out: dict[int, Phase1PartSummary] = {}
    for part in (1, 2, 3):
        part_rows = sorted((r for r in rows if r["part"] == part), key=lambda r: r["layer"])
        if not part_rows:
            continue
        peak = max(part_rows, key=lambda r: r["r2_loocv"])
        final = part_rows[-1]
        relative_drop = ((peak["r2_loocv"] - final["r2_loocv"]) / peak["r2_loocv"]
                         if peak["r2_loocv"] else None)
        out[part] = {
            "n_layers": len(part_rows), "best_r2": peak["r2_loocv"], "peak_layer": peak["layer"],
            "peak_perm_pvalue": peak["perm_pvalue"], "final_r2": final["r2_loocv"],
            "relative_drop": relative_drop,
        }
    return out


def _steering_category(direction: str, degradation: str) -> str:
    if direction == "band_probe":
        return "band_probe"
    is_orth = direction.endswith("_orth")
    base = direction[:-len("_orth")] if is_orth else direction
    matches = base == degradation
    if is_orth:
        return "self_orth" if matches else "other_orth"
    return "self" if matches else "other"


def phase3_category_summary(
    registry: "Registry", model_hf_id: str, run_scope: RunScope | None = None,
) -> dict[float, dict[str, float]]:
    where, params = _table_query_where(model_hf_id, "steer", run_scope)
    rows = registry.query(
        "SELECT s.alpha, s.direction, s.degradation, s.delta FROM steering_metrics s"
        " JOIN experiment_runs r ON r.id = s.run_id WHERE 1=1" + where,
        params)
    by_alpha: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        category = _steering_category(row["direction"], row["degradation"])
        by_alpha.setdefault(row["alpha"], {}).setdefault(category, []).append(row["delta"])
    return {alpha: {cat: st.mean(values) for cat, values in cats.items()}
           for alpha, cats in sorted(by_alpha.items())}


class SimilaritySummary(TypedDict):
    n: int
    mean: float | None
    median: float | None


def _summarize(values: list[float]) -> SimilaritySummary:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": st.mean(values), "median": st.median(values)}


def _reliable_rows(registry: "Registry", model_hf_id: str,
                   reliability_threshold: float,
                   run_scope: RunScope | None = None) -> list[dict]:
    where, params = _table_query_where(model_hf_id, "geometry", run_scope)
    query = (
        "SELECT cg.layer, cg.part_a, cg.degradation_a, cg.part_b, cg.degradation_b,"
        " cg.cosine_sim, cg.reliability_a, cg.reliability_b"
        " FROM contrastive_geometry cg JOIN experiment_runs r ON r.id = cg.run_id"
        " WHERE 1=1" + where
    )
    rows = registry.query(query, params)
    return [row for row in rows
           if (row["reliability_a"] or 0) >= reliability_threshold
           and (row["reliability_b"] or 0) >= reliability_threshold]


def _group_reliable_rows_by_pair(all_reliable: list[dict]) -> dict[tuple[str, str], list[dict]]:
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in all_reliable:
        key = tuple(sorted((row["degradation_a"], row["degradation_b"])))
        by_pair.setdefault(key, []).append(row)
    return by_pair


def _pair_disattenuation_triples(
    rows_for_pair: list[dict], criterion_a: str, criterion_b: str,
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for row in rows_for_pair:
        if row["degradation_a"] == criterion_a:
            rel_a, rel_b = row["reliability_a"], row["reliability_b"]
        else:
            rel_a, rel_b = row["reliability_b"], row["reliability_a"]
        out.append((row["cosine_sim"], rel_a, rel_b))
    return out


def _disattenuate(triples: list[tuple[float, float, float]]) -> dict:
    cos_values = [t[0] for t in triples]
    rel_a_values = [t[1] for t in triples]
    rel_b_values = [t[2] for t in triples]
    if not cos_values:
        return {
            "reliability_a_mean": None, "reliability_b_mean": None,
            "disattenuated_cosine_sim": None, "was_clipped": None,
            "gap_from_identity": None,
        }
    mean_cos = st.mean(cos_values)
    reliability_a_mean = st.mean(rel_a_values)
    reliability_b_mean = st.mean(rel_b_values)
    denom = (math.sqrt(reliability_a_mean * reliability_b_mean)
             if reliability_a_mean > 0 and reliability_b_mean > 0 else None)
    raw_ratio = (mean_cos / denom) if denom else None
    was_clipped = raw_ratio is not None and not (-1.0 <= raw_ratio <= 1.0)
    disattenuated_cosine_sim = (max(-1.0, min(1.0, raw_ratio))
                                if raw_ratio is not None else None)
    return {
        "reliability_a_mean": reliability_a_mean,
        "reliability_b_mean": reliability_b_mean,
        "disattenuated_cosine_sim": disattenuated_cosine_sim,
        "was_clipped": was_clipped,
        "gap_from_identity": (1 - disattenuated_cosine_sim
                              if disattenuated_cosine_sim is not None else None),
    }


def _pair_part_tier(rows_for_pair: list[dict]) -> dict:
    same_part_rows = [r for r in rows_for_pair if r["part_a"] == r["part_b"]]
    if same_part_rows:
        return {"primary_rows": same_part_rows, "tier": "same_part_verified"}
    return {"primary_rows": [], "tier": "not_comparable"}


def rq2_criterion_pair_gap_rows(
    registry: "Registry", model_hf_ids: list[str],
    reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
    run_scope: RunScope | None = None,
) -> list[dict]:
    all_reliable: list[dict] = []
    for model_hf_id in model_hf_ids:
        all_reliable.extend(_reliable_rows(registry, model_hf_id, reliability_threshold, run_scope))

    by_pair = _group_reliable_rows_by_pair(all_reliable)

    out: list[dict] = []
    for (criterion_a, criterion_b), rows_for_pair in sorted(by_pair.items()):
        is_same_criterion = criterion_a == criterion_b
        if is_same_criterion:
            primary_rows = rows_for_pair
            part_tier = None
        else:
            tiered = _pair_part_tier(rows_for_pair)
            primary_rows = tiered["primary_rows"]
            part_tier = tiered["tier"]

        values = [row["cosine_sim"] for row in primary_rows]
        mean_value = st.mean(values) if values else None
        triples = _pair_disattenuation_triples(primary_rows, criterion_a, criterion_b)
        disattenuation = _disattenuate(triples)

        out.append({
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "pair_type": "same_criterion" if is_same_criterion else "different_criterion",
            "part_tier": part_tier,
            "n": len(values),
            "mean_cosine_sim": mean_value,
            **disattenuation,
            "reliability_threshold": reliability_threshold,
        })
    return out


def rq2_criterion_pair_gap_significance_rows(
    registry: "Registry", model_hf_ids: list[str],
    reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
    run_scope: RunScope | None = None,
    n_bootstrap: int = DEFAULT_RESAMPLES,
    seed: int = 1729,
) -> list[dict]:
    import numpy as np

    all_reliable: list[dict] = []
    for model_hf_id in model_hf_ids:
        all_reliable.extend(_reliable_rows(registry, model_hf_id, reliability_threshold, run_scope))
    by_pair = _group_reliable_rows_by_pair(all_reliable)

    rng = np.random.default_rng(seed)

    out: list[dict] = []
    for (criterion_a, criterion_b), rows_for_pair in sorted(by_pair.items()):
        is_same_criterion = criterion_a == criterion_b
        if is_same_criterion:
            primary_rows = rows_for_pair
            part_tier = None
        else:
            tiered = _pair_part_tier(rows_for_pair)
            primary_rows = tiered["primary_rows"]
            part_tier = tiered["tier"]

        triples = _pair_disattenuation_triples(primary_rows, criterion_a, criterion_b)
        n = len(triples)
        mean_value = float(np.mean([t[0] for t in triples])) if n else None
        point_estimate = _disattenuate(triples)
        row_out = {
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "pair_type": "same_criterion" if is_same_criterion else "different_criterion",
            "part_tier": part_tier,
            "n": n,
            "mean_cosine_sim": mean_value,
            "disattenuated_cosine_sim": point_estimate["disattenuated_cosine_sim"],
            "gap_from_identity": point_estimate["gap_from_identity"],
            "ci_low_disattenuated": None,
            "ci_high_disattenuated": None,
            "distinguishable_from_identity": None,
        }
        if n < 2:
            out.append(row_out)
            continue
        cos_arr = np.asarray([t[0] for t in triples], dtype=float)
        chunk = max(1, min(n_bootstrap, 5_000_000 // max(1, n)))
        disattenuated_draws: list[float] = []
        rel_a_arr = np.asarray([t[1] for t in triples], dtype=float)
        rel_b_arr = np.asarray([t[2] for t in triples], dtype=float)
        for start in range(0, n_bootstrap, chunk):
            size = min(chunk, n_bootstrap - start)
            idx = rng.integers(0, n, size=(size, n))
            boot_cos = cos_arr[idx].mean(axis=1)
            boot_rel_a = rel_a_arr[idx].mean(axis=1)
            boot_rel_b = rel_b_arr[idx].mean(axis=1)
            boot_denom = np.sqrt(boot_rel_a * boot_rel_b)
            boot_raw = np.divide(boot_cos, boot_denom, out=np.zeros_like(boot_cos),
                                  where=boot_denom != 0)
            boot_disattenuated = np.clip(boot_raw, -1.0, 1.0)
            disattenuated_draws.extend(boot_disattenuated.tolist())

        ci_low_dis, ci_high_dis = _ci(sorted(disattenuated_draws))
        row_out["ci_low_disattenuated"] = ci_low_dis
        row_out["ci_high_disattenuated"] = ci_high_dis
        row_out["distinguishable_from_identity"] = (
            ci_high_dis is not None and ci_high_dis < 1.0)
        out.append(row_out)
    return out


def steering_target_candidate_lookup(registry: "Registry") -> dict[tuple[int, int], str]:
    rows = registry.query(
        "SELECT a.path FROM artifacts a JOIN experiment_runs r ON r.id = a.run_id"
        " WHERE a.kind = 'mutated_text' ORDER BY r.started_at DESC LIMIT 1")
    if not rows:
        return {}
    path = Path(rows[0]["path"])
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, int], str] = {}
    for record in records:
        if "candidate_id" not in record:
            continue
        out.setdefault((int(record["form"]), int(record["part"])), str(record["candidate_id"]))
    return out


def phase3_self_other_paired_rows(
    registry: "Registry", model_hf_ids: list[str], run_scope: RunScope | None = None,
) -> list[dict]:
    out: list[dict] = []
    candidate_by_form_part = steering_target_candidate_lookup(registry)
    for model_hf_id in model_hf_ids:
        where, params = _table_query_where(model_hf_id, "steer", run_scope)
        rows = registry.query(
            "SELECT s.layer, s.part, s.form, s.direction, s.degradation, s.alpha, s.delta"
            " FROM steering_metrics s JOIN experiment_runs r ON r.id = s.run_id"
            " WHERE 1=1" + where,
            params)
        by_key: dict[tuple[int, int, int, str, float], dict[str, list[float]]] = {}
        for row in rows:
            direction = row["direction"]
            if direction == "band_probe" or direction.endswith("_orth"):
                continue
            key = (row["layer"], row["part"], row["form"], row["degradation"], row["alpha"])
            bucket = "self" if direction == row["degradation"] else "other"
            by_key.setdefault(key, {"self": [], "other": []})[bucket].append(row["delta"])
        for (layer, part, form, degradation, alpha), vals in sorted(by_key.items()):
            if not vals["self"] or not vals["other"]:
                continue
            self_mean = st.mean(vals["self"])
            other_mean = st.mean(vals["other"])
            gap = self_mean - other_mean
            abs_self = abs(self_mean)
            out.append({
                "model_hf_id": model_hf_id,
                "layer": layer,
                "part": part,
                "form": form,
                "candidate_id_inferred": candidate_by_form_part.get((form, part)),
                "degradation": degradation,
                "alpha": alpha,
                "self_delta_mean": self_mean,
                "other_delta_mean": other_mean,
                "self_other_gap": gap,
                "abs_self_delta_mean": abs_self,
                "specificity_ratio": gap / abs_self if abs_self > 1e-12 else None,
                "n_self_rows": len(vals["self"]),
                "n_other_rows": len(vals["other"]),
            })
    return out


def phase3_specificity_ratio_by_model_rows(paired_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in paired_rows:
        group = grouped.setdefault(row["model_hf_id"], {"self": [], "other": [], "gap": []})
        group["self"].append(row["self_delta_mean"])
        group["other"].append(row["other_delta_mean"])
        group["gap"].append(row["self_other_gap"])
    out: list[dict] = []
    for model_hf_id, vals in sorted(grouped.items()):
        gap_ci_lo, gap_ci_hi = _bootstrap_mean_ci(vals["gap"])
        mean_self = st.mean(vals["self"])
        mean_other = st.mean(vals["other"])
        mean_gap = st.mean(vals["gap"])
        mean_abs_self = st.mean(abs(v) for v in vals["self"])
        out.append({
            "model_hf_id": model_hf_id,
            "n_paired_cells": len(vals["gap"]),
            "self_delta_mean": mean_self,
            "other_delta_mean": mean_other,
            "self_other_gap_mean": mean_gap,
            "self_other_gap_ci_low": gap_ci_lo,
            "self_other_gap_ci_high": gap_ci_hi,
            "abs_self_delta_mean": mean_abs_self,
            "specificity_ratio": mean_gap / mean_abs_self if mean_abs_self > 1e-12 else None,
        })
    return out


def phase3_equivalence_summary(
    paired_rows: list[dict],
    model_summary_rows: list[dict],
    margin_score_points: float = 0.05,
    ratio_margin: float = 0.10,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    model_gaps = [r["self_other_gap_mean"] for r in model_summary_rows]
    model_ratios = [r["specificity_ratio"] for r in model_summary_rows
                    if r["specificity_ratio"] is not None]
    cell_gaps = [r["self_other_gap"] for r in paired_rows]
    mean_model_gap = st.mean(model_gaps) if model_gaps else None
    model_ci_low, model_ci_high = _bootstrap_mean_ci(model_gaps, n_resamples)
    cell_ci_low, cell_ci_high = _bootstrap_mean_ci(cell_gaps, n_resamples)
    mean_ratio = st.mean(model_ratios) if model_ratios else None
    ratio_ci_low, ratio_ci_high = _bootstrap_mean_ci(model_ratios, n_resamples)
    return {
        "n_resamples": n_resamples,
        "margin_score_points": margin_score_points,
        "ratio_margin": ratio_margin,
        "n_models": len(model_gaps),
        "n_paired_cells": len(cell_gaps),
        "model_level_gap_mean": mean_model_gap,
        "model_level_gap_ci_low": model_ci_low,
        "model_level_gap_ci_high": model_ci_high,
        "model_level_equivalent_within_margin": (
            model_ci_low is not None and model_ci_high is not None
            and model_ci_low > -margin_score_points and model_ci_high < margin_score_points),
        "cell_level_gap_mean": st.mean(cell_gaps) if cell_gaps else None,
        "cell_level_gap_ci_low": cell_ci_low,
        "cell_level_gap_ci_high": cell_ci_high,
        "cell_level_equivalent_within_margin": (
            cell_ci_low is not None and cell_ci_high is not None
            and cell_ci_low > -margin_score_points and cell_ci_high < margin_score_points),
        "model_level_specificity_ratio_mean": mean_ratio,
        "model_level_specificity_ratio_ci_low": ratio_ci_low,
        "model_level_specificity_ratio_ci_high": ratio_ci_high,
        "model_level_ratio_equivalent_within_margin": (
            ratio_ci_low is not None and ratio_ci_high is not None
            and ratio_ci_low > -ratio_margin and ratio_ci_high < ratio_margin),
    }


def phase3_per_pair_equivalence_rows(
    registry: "Registry", model_hf_ids: list[str],
    run_scope: RunScope | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 1729,
) -> list[dict]:
    self_by_model: dict[str, dict[str, list[float]]] = {}
    other_by_model: dict[str, dict[tuple[str, str], list[float]]] = {}
    for model_hf_id in model_hf_ids:
        where, params = _table_query_where(model_hf_id, "steer", run_scope)
        rows = registry.query(
            "SELECT s.direction, s.degradation, s.delta"
            " FROM steering_metrics s JOIN experiment_runs r ON r.id = s.run_id"
            " WHERE 1=1" + where, params)
        self_vals = self_by_model.setdefault(model_hf_id, {})
        other_vals = other_by_model.setdefault(model_hf_id, {})
        for row in rows:
            direction = row["direction"]
            if direction == "band_probe" or direction.endswith("_orth"):
                continue
            degradation = row["degradation"]
            if direction == degradation:
                self_vals.setdefault(degradation, []).append(row["delta"])
            else:
                other_vals.setdefault((degradation, direction), []).append(row["delta"])

    self_effect_summary: dict[str, dict] = {}
    self_criteria = sorted({c for vals in self_by_model.values() for c in vals})
    for self_criterion in self_criteria:
        self_model_means = []
        for model_hf_id in model_hf_ids:
            self_list = self_by_model.get(model_hf_id, {}).get(self_criterion)
            if self_list:
                self_model_means.append(st.mean(self_list))
        if len(self_model_means) < 3:
            continue
        mean_self = st.mean(self_model_means)
        self_ci_low, self_ci_high = _bootstrap_mean_ci(self_model_means, n_resamples, seed)
        self_effect_summary[self_criterion] = {
            "self_effect_mean": mean_self,
            "self_effect_ci_low": self_ci_low,
            "self_effect_ci_high": self_ci_high,
            "self_effect_distinguishable_from_zero": (
                self_ci_low is not None and self_ci_high is not None
                and (self_ci_low > 0 or self_ci_high < 0)),
        }

    all_pairs = sorted({pair for vals in other_by_model.values() for pair in vals})
    out: list[dict] = []
    for self_criterion, other_criterion in all_pairs:
        model_gaps = []
        other_model_means = []
        for model_hf_id in model_hf_ids:
            self_list = self_by_model.get(model_hf_id, {}).get(self_criterion)
            other_list = other_by_model.get(model_hf_id, {}).get((self_criterion, other_criterion))
            if not self_list or not other_list:
                continue
            model_gaps.append(st.mean(self_list) - st.mean(other_list))
            other_model_means.append(st.mean(other_list))
        if len(model_gaps) < 3:
            continue
        mean_gap = st.mean(model_gaps)
        ci_low, ci_high = _bootstrap_mean_ci(model_gaps, n_resamples, seed)
        out.append({
            "self_criterion": self_criterion,
            "other_criterion": other_criterion,
            "n_models": len(model_gaps),
            **self_effect_summary.get(self_criterion, {
                "self_effect_mean": None, "self_effect_ci_low": None,
                "self_effect_ci_high": None,
                "self_effect_distinguishable_from_zero": False,
            }),
            "other_effect_mean": st.mean(other_model_means),
            "model_level_gap_mean": mean_gap,
            "model_level_gap_ci_low": ci_low,
            "model_level_gap_ci_high": ci_high,
            "distinguishable_from_zero": (
                ci_low is not None and ci_high is not None
                and (ci_low > 0 or ci_high < 0)),
        })
    return out


def phase3_alpha_slope_rows(paired_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[float, dict[str, list[float]]]] = {}
    for row in paired_rows:
        by_alpha = grouped.setdefault(row["model_hf_id"], {})
        vals = by_alpha.setdefault(row["alpha"], {"self": [], "other": [], "gap": []})
        vals["self"].append(row["self_delta_mean"])
        vals["other"].append(row["other_delta_mean"])
        vals["gap"].append(row["self_other_gap"])
    out: list[dict] = []
    for model_hf_id, by_alpha in sorted(grouped.items()):
        alphas = sorted(by_alpha)
        self_means = [st.mean(by_alpha[a]["self"]) for a in alphas]
        other_means = [st.mean(by_alpha[a]["other"]) for a in alphas]
        gap_means = [st.mean(by_alpha[a]["gap"]) for a in alphas]
        self_fit = _ols_slope_intercept(alphas, self_means)
        other_fit = _ols_slope_intercept(alphas, other_means)
        gap_fit = _ols_slope_intercept(alphas, gap_means)
        out.append({
            "model_hf_id": model_hf_id,
            "n_alpha_levels": len(alphas),
            "self_alpha_slope": self_fit["slope"],
            "other_alpha_slope": other_fit["slope"],
            "gap_alpha_slope": gap_fit["slope"],
            "self_alpha_r": self_fit["r"],
            "other_alpha_r": other_fit["r"],
            "gap_alpha_r": gap_fit["r"],
            "alpha_slope_difference_self_minus_other": (
                self_fit["slope"] - other_fit["slope"]
                if self_fit["slope"] is not None and other_fit["slope"] is not None else None),
        })
    return out


def phase3_alpha_summary(alpha_rows: list[dict], n_resamples: int = DEFAULT_RESAMPLES) -> dict:
    self_slopes = [r["self_alpha_slope"] for r in alpha_rows if r["self_alpha_slope"] is not None]
    other_slopes = [r["other_alpha_slope"] for r in alpha_rows if r["other_alpha_slope"] is not None]
    gap_slopes = [r["gap_alpha_slope"] for r in alpha_rows if r["gap_alpha_slope"] is not None]
    return {
        "n_models": len(alpha_rows),
        "n_resamples": n_resamples,
        "self_alpha_slope_mean": st.mean(self_slopes) if self_slopes else None,
        "other_alpha_slope_mean": st.mean(other_slopes) if other_slopes else None,
        "gap_alpha_slope_mean": st.mean(gap_slopes) if gap_slopes else None,
        "self_alpha_slope_ci_low": _bootstrap_mean_ci(self_slopes, n_resamples)[0],
        "self_alpha_slope_ci_high": _bootstrap_mean_ci(self_slopes, n_resamples)[1],
        "other_alpha_slope_ci_low": _bootstrap_mean_ci(other_slopes, n_resamples)[0],
        "other_alpha_slope_ci_high": _bootstrap_mean_ci(other_slopes, n_resamples)[1],
        "gap_alpha_slope_ci_low": _bootstrap_mean_ci(gap_slopes, n_resamples)[0],
        "gap_alpha_slope_ci_high": _bootstrap_mean_ci(gap_slopes, n_resamples)[1],
    }


def _safe_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return None
    try:
        return st.correlation(xs, ys)
    except st.StatisticsError:
        return None


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _ci(values: list[float], lo: float = 0.025, hi: float = 0.975) -> tuple[float | None, float | None]:
    ordered = sorted(values)
    return _quantile(ordered, lo), _quantile(ordered, hi)


def _bootstrap_mean_ci(
    values: list[float], n_bootstrap: int = DEFAULT_RESAMPLES, seed: int = 1729,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(values)
    chunk = max(1, min(n_bootstrap, 5_000_000 // max(1, n)))
    draws: list[float] = []
    for start in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        draws.extend(arr[indices].mean(axis=1).tolist())
    return _ci(draws)


def _ols_slope_intercept(xs: list[float], ys: list[float]) -> dict:
    if len(xs) < 2 or len(xs) != len(ys):
        return {"n": len(xs), "slope": None, "intercept": None, "r": None}
    try:
        slope, intercept = st.linear_regression(xs, ys)
    except st.StatisticsError:
        return {"n": len(xs), "slope": None, "intercept": None, "r": None}
    return {
        "n": len(xs),
        "slope": slope,
        "intercept": intercept,
        "r": _safe_correlation(xs, ys),
    }


# ---- Camera-ready additions (RQ1 permutation baseline, RQ3 pairwise check) ----

def _pairwise_steering_category(direction: str, degradation: str) -> str | None:
    """Recognizes the `{a}_uniqueto_{b}` / `{a}_uniqueto_{b}_randnoise` labels
    written by the pairwise self/other-unique orthogonalization block added
    to `_phase_steer` (STATUS.md 2026-08-15 RQ3 PI-feedback refinement).
    Returns None for any other direction label (`band_probe`, raw criterion
    names, `_orth`) — this is a DIFFERENT taxonomy from `_steering_category`,
    not an extension of it, since `_uniqueto_` never collides with those
    (kept as a separate function rather than folding into `_steering_category`
    to avoid a criterion name accidentally substring-matching `_orth`-style
    parsing there — see the callers of `_steering_category`, none of which
    expect this new label shape)."""
    is_noise = direction.endswith("_randnoise")
    body = direction[: -len("_randnoise")] if is_noise else direction
    if "_uniqueto_" not in body:
        return None
    prefix, _, _other = body.partition("_uniqueto_")
    if is_noise:
        return "noise_floor"
    return "self_unique" if prefix == degradation else "other_unique"


def phase3_pairwise_specificity_rows(
    registry: "Registry", model_hf_id: str, run_scope: RunScope | None = None,
) -> list[dict]:
    """Per-(part, self_degradation, other_degradation) self_unique vs
    other_unique vs noise_floor means, mean-collapsed across layers/alphas/
    forms — the number that actually answers the PI's construct-overlap
    concern (STATUS.md 2026-08-15): does `self_unique`'s steering effect
    clear `noise_floor` while `other_unique`'s does not (real
    criterion-specific residual survives worst-case correction), or is the
    observed self-vs-other gap itself indistinguishable from what pure noise
    at that same residual magnitude produces (confirms the "interchangeable
    levers" finding is a noise/overlap artifact)? Each row also carries the
    pair's own disattenuated cosine (Table 4) from `contrastive_geometry` so
    near-ceiling pairs — inconclusive by construction, not evidence either
    way — can be filtered before interpreting the rest.

    Caveat on `run_scope`: because `_phase_steer`'s idempotency guard lets a
    re-run add pairwise rows under a NEW run_id while skipping (not
    rewriting) already-computed raw/orth rows under the ORIGINAL run_id, a
    single pinned `steer_run_id` (as `model_set.csv` / `RunScope` normally
    freeze for reproducible paper numbers) will not see both halves at once.
    Leave `run_scope=None` (the default — aggregates by `model_hf_id` across
    every run, matching how `_latest_activations` already resolves "latest
    for this model" rather than "latest for this run") until the model's
    `RunScope` entry is deliberately regenerated to reflect this."""
    where, params = _table_query_where(model_hf_id, "steer", run_scope)
    rows = registry.query(
        "SELECT s.part, s.layer, s.direction, s.degradation, s.delta"
        " FROM steering_metrics s JOIN experiment_runs r ON r.id = s.run_id"
        " WHERE 1=1" + where, params)
    by_pair: dict[tuple[int, str, str], dict[str, list[float]]] = {}
    for row in rows:
        category = _pairwise_steering_category(row["direction"], row["degradation"])
        if category is None:
            continue
        body = row["direction"]
        if body.endswith("_randnoise"):
            body = body[: -len("_randnoise")]
        prefix, _, suffix = body.partition("_uniqueto_")
        degradation = row["degradation"]
        # the noise-floor label is `{other}_uniqueto_{self}_randnoise` -- its
        # PREFIX is other_degradation, unlike the two non-noise labels whose
        # prefix is self_degradation -- so identify self/other by whichever
        # of {prefix, suffix} actually equals the row's own degradation,
        # not by position. (A bug here once mapped every noise-floor row to
        # a spurious self==other pair -- see the regression test.)
        if prefix == degradation:
            self_degradation, other_degradation = degradation, suffix
        elif suffix == degradation:
            self_degradation, other_degradation = degradation, prefix
        else:
            continue  # malformed direction label; shouldn't happen
        key = (row["part"], self_degradation, other_degradation)
        by_pair.setdefault(key, {}).setdefault(category, []).append(row["delta"])

    cg_where, cg_params = _table_query_where(model_hf_id, "geometry", run_scope)
    cosine_rows = registry.query(
        "SELECT cg.part_a, cg.degradation_a, cg.part_b, cg.degradation_b,"
        " cg.cosine_sim, cg.reliability_a, cg.reliability_b"
        " FROM contrastive_geometry cg JOIN experiment_runs r ON r.id = cg.run_id"
        " WHERE cg.part_a = cg.part_b" + cg_where, cg_params)
    cosines: dict[tuple[int, str, str], float] = {}
    for row in cosine_rows:
        ra, rb = row["reliability_a"], row["reliability_b"]
        cos = row["cosine_sim"]
        disattenuated = cos / (ra * rb) ** 0.5 if ra and rb and ra > 0 and rb > 0 else cos
        cosines[(row["part_a"], row["degradation_a"], row["degradation_b"])] = (
            max(-1.0, min(1.0, disattenuated)))

    out: list[dict] = []
    for (part, self_degradation, other_degradation), cats in sorted(by_pair.items()):
        self_values = cats.get("self_unique", [])
        other_values = cats.get("other_unique", [])
        noise_values = cats.get("noise_floor", [])
        deg_a, deg_b = sorted((self_degradation, other_degradation))
        self_mean = st.mean(self_values) if self_values else None
        other_mean = st.mean(other_values) if other_values else None
        noise_mean = st.mean(noise_values) if noise_values else None
        out.append({
            "part": part,
            "self_degradation": self_degradation,
            "other_degradation": other_degradation,
            "disattenuated_cosine": cosines.get((part, deg_a, deg_b)),
            "n_self_unique": len(self_values),
            "n_other_unique": len(other_values),
            "n_noise_floor": len(noise_values),
            "self_unique_mean": self_mean,
            "other_unique_mean": other_mean,
            "noise_floor_mean": noise_mean,
            "self_unique_minus_noise": (
                self_mean - noise_mean if self_mean is not None and noise_mean is not None
                else None),
            "other_unique_minus_noise": (
                other_mean - noise_mean if other_mean is not None and noise_mean is not None
                else None),
            "self_other_unique_gap": (
                self_mean - other_mean if self_mean is not None and other_mean is not None
                else None),
        })
    return out


def phase3_pairwise_specificity_all_models(
    registry: "Registry", model_hf_ids: list[str], run_scope: RunScope | None = None,
) -> list[dict]:
    """Flattens `phase3_pairwise_specificity_rows` across `model_hf_ids`,
    tagging each row with `model_hf_id` -- the cross-model input
    `phase3_pairwise_specificity_pooled_rows` expects. Models with no
    pairwise (`_uniqueto_`) data yet (steer run predates the pairwise
    backfill, or hasn't reached this model) simply contribute zero rows,
    same silently-partial-coverage convention as `phase3_self_other_paired_rows`."""
    out: list[dict] = []
    for model_hf_id in model_hf_ids:
        for row in phase3_pairwise_specificity_rows(registry, model_hf_id, run_scope):
            tagged = dict(row)
            tagged["model_hf_id"] = model_hf_id
            out.append(tagged)
    return out


def phase3_pairwise_specificity_pooled_rows(
    paired_rows: list[dict], n_resamples: int = DEFAULT_RESAMPLES,
) -> list[dict]:
    """Pools `phase3_pairwise_specificity_all_models` output into the actual
    answer to the PI's construct-overlap concern (STATUS.md 2026-08-15): does
    `self_unique`'s steering effect exceed `other_unique`'s by more than the
    random-direction noise floor, once cells are restricted to pairs whose
    disattenuated cosine similarity (Table 4) is low/moderate enough for the
    orthogonalized residual to still carry real signal rather than being
    mostly noise by construction?

    Reuses `DEFAULT_RELIABILITY_THRESHOLD` (0.6) as the cosine cutoff between
    the `trustworthy` and `near_ceiling` subsets -- the same bar this module
    already uses everywhere else to mean "clears the floor for a meaningful
    comparison," not a new threshold invented for this check. `near_ceiling`
    pairs are still reported (never dropped) so a reader can confirm they
    behave as expected under the "inconclusive by construction" framing
    (small `self_unique`/`other_unique` values, since most of the true
    signal was projected away by construction) rather than trusting that
    framing on faith.

    Each subset's `self_other_unique_gap` (self_unique minus other_unique)
    is the actual specificity test; `self/other_unique_minus_noise` confirm
    the orthogonalized residual carries real signal above pure noise before
    that gap is interpreted at all -- mirrors the resampling and
    specificity-ratio conventions of `phase3_reliability_gated_specificity_rows`."""
    complete = [
        r for r in paired_rows
        if r["self_unique_mean"] is not None and r["other_unique_mean"] is not None
        and r["noise_floor_mean"] is not None
    ]
    subset_specs = [
        ("all_pairs", complete),
        ("trustworthy_cosine_below_0_6",
         [r for r in complete if r["disattenuated_cosine"] is not None
          and r["disattenuated_cosine"] < DEFAULT_RELIABILITY_THRESHOLD]),
        ("near_ceiling_cosine_ge_0_6",
         [r for r in complete if r["disattenuated_cosine"] is not None
          and r["disattenuated_cosine"] >= DEFAULT_RELIABILITY_THRESHOLD]),
    ]
    out: list[dict] = []
    for label, rows in subset_specs:
        gaps = [r["self_other_unique_gap"] for r in rows]
        self_minus_noise = [r["self_unique_minus_noise"] for r in rows]
        other_minus_noise = [r["other_unique_minus_noise"] for r in rows]
        abs_self = [abs(r["self_unique_mean"]) for r in rows]
        mean_gap = st.mean(gaps) if gaps else None
        mean_abs_self = st.mean(abs_self) if abs_self else None
        gap_ci_low, gap_ci_high = _bootstrap_mean_ci(gaps, n_resamples) if gaps else (None, None)
        smn_ci_low, smn_ci_high = (
            _bootstrap_mean_ci(self_minus_noise, n_resamples) if self_minus_noise
            else (None, None))
        omn_ci_low, omn_ci_high = (
            _bootstrap_mean_ci(other_minus_noise, n_resamples) if other_minus_noise
            else (None, None))
        out.append({
            "subset": label,
            "n_resamples": n_resamples,
            "n_cells": len(rows),
            "n_models": len({r["model_hf_id"] for r in rows}),
            "self_unique_mean": st.mean(r["self_unique_mean"] for r in rows) if rows else None,
            "other_unique_mean": st.mean(r["other_unique_mean"] for r in rows) if rows else None,
            "noise_floor_mean": st.mean(r["noise_floor_mean"] for r in rows) if rows else None,
            "self_unique_minus_noise_mean": st.mean(self_minus_noise) if self_minus_noise else None,
            "self_unique_minus_noise_ci_low": smn_ci_low,
            "self_unique_minus_noise_ci_high": smn_ci_high,
            "other_unique_minus_noise_mean": st.mean(other_minus_noise) if other_minus_noise else None,
            "other_unique_minus_noise_ci_low": omn_ci_low,
            "other_unique_minus_noise_ci_high": omn_ci_high,
            "self_other_unique_gap_mean": mean_gap,
            "self_other_unique_gap_ci_low": gap_ci_low,
            "self_other_unique_gap_ci_high": gap_ci_high,
            "specificity_ratio": (
                mean_gap / mean_abs_self if mean_gap is not None and mean_abs_self
                and mean_abs_self > 1e-12 else None),
        })
    return out


def rq1_layer_part_rows(
    registry: "Registry", model_hf_ids: list[str], run_scope: RunScope | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for model_hf_id in model_hf_ids:
        where, params = _table_query_where(model_hf_id, "probe", run_scope)
        db_rows = registry.query(
            "SELECT p.part, p.layer, p.r2_loocv, p.perm_pvalue FROM probes p"
            " JOIN experiment_runs r ON r.id = p.run_id"
            " WHERE p.form_scope = 'pooled'" + where,
            params)
        by_part: dict[int, list[dict]] = {}
        for row in db_rows:
            by_part.setdefault(row["part"], []).append(dict(row))
        for part, part_rows in by_part.items():
            ordered = sorted(part_rows, key=lambda row: row["layer"])
            if not ordered:
                continue
            peak = max(ordered, key=lambda row: row["r2_loocv"])
            final = ordered[-1]
            r2_values = [row["r2_loocv"] for row in ordered]
            r2_min, r2_max = min(r2_values), max(r2_values)
            denom = r2_max - r2_min
            for row in ordered:
                rows.append({
                    "model_hf_id": model_hf_id,
                    "part": part,
                    "layer": row["layer"],
                    "r2_loocv": row["r2_loocv"],
                    "perm_pvalue": row["perm_pvalue"],
                    "peak_layer": peak["layer"],
                    "final_layer": final["layer"],
                    "distance_from_peak": row["layer"] - peak["layer"],
                    "abs_distance_from_peak": abs(row["layer"] - peak["layer"]),
                    "is_peak_layer": row["layer"] == peak["layer"],
                    "is_after_peak": row["layer"] > peak["layer"],
                    "normalized_r2_within_part": ((row["r2_loocv"] - r2_min) / denom
                                                  if denom else None),
                    "peak_r2": peak["r2_loocv"],
                    "final_r2": final["r2_loocv"],
                    "relative_drop": ((peak["r2_loocv"] - final["r2_loocv"]) / peak["r2_loocv"]
                                      if peak["r2_loocv"] else None),
                })
    return rows


def rq1_peak_permutation_rows(rq1_rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """One row per (model, part): the peak layer's stored label-shuffle permutation
    p-value, plus that p-value Bonferroni-corrected over the model's layers.

    The peak layer is the best of `n_layers` layers, so its raw per-layer p-value
    ignores that selection; the layer-corrected value `min(1, p * n_layers)` is the
    honest one. Bonferroni is conservative (adjacent layers are correlated), and
    the 1000-shuffle floor of p = 1/1001 bounds how small the corrected value can
    get at large `n_layers`. A max-over-layers permutation test would be exact but
    needs the cached activations. Pure function of `rq1_layer_part_rows` output, so
    it works for any cohort."""
    n_layers: dict[tuple[str, int], int] = {}
    for row in rq1_rows:
        key = (row["model_hf_id"], row["part"])
        n_layers[key] = n_layers.get(key, 0) + 1
    out: list[dict] = []
    for row in rq1_rows:
        if not row["is_peak_layer"]:
            continue
        p_raw = row["perm_pvalue"]
        n = n_layers[(row["model_hf_id"], row["part"])]
        p_adj = None if p_raw is None else min(1.0, p_raw * n)
        out.append({
            "model_hf_id": row["model_hf_id"],
            "part": row["part"],
            "peak_layer": row["layer"],
            "n_layers": n,
            "peak_r2": row["r2_loocv"],
            "perm_pvalue_raw": p_raw,
            "perm_pvalue_layer_bonferroni": p_adj,
            "sig_raw": None if p_raw is None else p_raw < alpha,
            "sig_layer_bonferroni": None if p_adj is None else p_adj < alpha,
        })
    return out


def rq1_peak_permutation_summary(perm_rows: list[dict]) -> list[dict]:
    """Per-part counts of models whose peak layer beats the label-shuffle null."""
    out: list[dict] = []
    for part in sorted({r["part"] for r in perm_rows}):
        rows = [r for r in perm_rows if r["part"] == part and r["perm_pvalue_raw"] is not None]
        out.append({
            "part": part,
            "n_models": len(rows),
            "median_peak_r2": st.median(r["peak_r2"] for r in rows) if rows else None,
            "n_sig_raw": sum(r["sig_raw"] for r in rows),
            "n_sig_layer_bonferroni": sum(r["sig_layer_bonferroni"] for r in rows),
        })
    return out
