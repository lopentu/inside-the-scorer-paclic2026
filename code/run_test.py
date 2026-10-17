from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import (
    build_run_audit,
    paper_model_ids,
    phase3_alpha_slope_rows,
    phase3_alpha_summary,
    phase3_category_summary,
    phase3_equivalence_summary,
    phase3_pairwise_specificity_all_models,
    phase3_pairwise_specificity_pooled_rows,
    phase3_per_pair_equivalence_rows,
    phase3_within_part_matching_advantage_rows,
    phase3_self_other_paired_rows,
    phase3_specificity_ratio_by_model_rows,
    rq1_layer_part_rows,
    rq1_peak_permutation_rows,
    rq1_peak_permutation_summary,
    rq2_criterion_pair_gap_rows,
    rq2_criterion_pair_gap_significance_rows,
)
from registry import Registry

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "bestep_25models.db"

CHECKS: list[tuple[str, object, object]] = []


def check(label: str, actual: object, expected: object, tol: float = 0.01) -> None:
    if isinstance(expected, (int,)) and not isinstance(expected, bool):
        ok = actual == expected
    else:
        ok = abs(actual - expected) <= tol
    CHECKS.append((label, actual, expected, ok))


def ensure_db() -> None:
    """The repo ships the database gzip-compressed (33 MB vs 273 MB); unpack it once."""
    if DB_PATH.exists():
        return
    import gzip
    import shutil

    packed = DB_PATH.with_name(DB_PATH.name + ".gz")
    if not packed.exists():
        sys.exit(f"missing {DB_PATH} and {packed}")
    print(f"unpacking {packed.name} -> {DB_PATH.name} (one-time, about 273 MB)")
    with gzip.open(packed, "rb") as src, open(DB_PATH, "wb") as dst:
        shutil.copyfileobj(src, dst)


def main() -> None:
    ensure_db()
    registry = Registry(DB_PATH)
    model_ids = paper_model_ids()
    run_scope, _ = build_run_audit(registry, model_ids)

    print("=== RQ2: Table 4 (disattenuated criterion-pair similarity) ===")
    sig_rows = rq2_criterion_pair_gap_significance_rows(registry, model_ids, run_scope=run_scope)
    diff_pairs = {(r["criterion_a"], r["criterion_b"]): r for r in sig_rows
                  if r["pair_type"] == "different_criterion" and r["part_tier"] != "not_comparable"}
    expected_table4 = {
        ("argument_relevance", "grammar"): (564, -0.10),
        ("spelling_punctuation", "task_relevance"): (6, 0.01),
        ("argument_relevance", "cohesion"): (586, 0.08),
        ("grammar", "task_relevance"): (416, 0.28),
        ("argument_relevance", "vocabulary"): (419, 0.31),
        ("cohesion", "grammar"): (800, 0.39),
        ("grammar", "vocabulary"): (823, 0.49),
        ("task_relevance", "vocabulary"): (336, 0.51),
        ("cohesion", "task_relevance"): (334, 0.52),
        ("cohesion", "vocabulary"): (691, 0.61),
        ("spelling_punctuation", "vocabulary"): (192, 0.98),
        ("grammar", "spelling_punctuation"): (328, 1.00),
    }
    for pair, (exp_n, exp_sim) in expected_table4.items():
        row = diff_pairs.get(pair) or diff_pairs.get((pair[1], pair[0]))
        assert row is not None, f"missing pair {pair}"
        check(f"Table4 n {pair}", row["n"], exp_n, tol=0)
        check(f"Table4 sim {pair}", row["disattenuated_cosine_sim"], exp_sim, tol=0.01)

    print("=== RQ3: self/other headline (74,304 paired cells) ===")
    paired_rows = phase3_self_other_paired_rows(registry, model_ids, run_scope)
    model_summary = phase3_specificity_ratio_by_model_rows(paired_rows)
    equivalence = phase3_equivalence_summary(paired_rows, model_summary)
    check("n_paired_cells", equivalence["n_paired_cells"], 74304, tol=0)
    check("cell_level_gap_mean", equivalence["cell_level_gap_mean"], 0.0085, tol=0.001)
    check("model_level_gap_mean", equivalence["model_level_gap_mean"], 0.0074, tol=0.001)

    print("=== RQ3: per-criterion self effects (0.168 to 0.359) ===")
    pair_rows = phase3_per_pair_equivalence_rows(registry, model_ids, run_scope=run_scope)
    self_effects = {}
    for row in pair_rows:
        self_effects.setdefault(row["self_criterion"], row["self_effect_mean"])
    expected_self = {
        "argument_relevance": -0.083, "cohesion": 0.229, "grammar": 0.312,
        "spelling_punctuation": 0.359, "task_relevance": 0.168, "vocabulary": 0.233,
    }
    for criterion, exp_val in expected_self.items():
        check(f"self_effect[{criterion}]", self_effects[criterion], exp_val, tol=0.01)

    print("=== RQ3: new Table 5 (orthogonal control / band_probe) ===")
    orth_deltas, bp_deltas, orth_slopes, bp_slopes = [], [], [], []
    orth_slopes, bp_slopes = [], []
    for model_id in model_ids:
        by_alpha = phase3_category_summary(registry, model_id, run_scope)
        oa, od, ba, bd = [], [], [], []
        for alpha, cats in by_alpha.items():
            vals = [cats[c] for c in ("self_orth", "other_orth") if c in cats]
            if vals:
                oa.append(alpha)
                od.append(st.mean(vals))
            if "band_probe" in cats:
                ba.append(alpha)
                bd.append(cats["band_probe"])
        if od:
            orth_deltas.append(st.mean(od))
            orth_slopes.append(st.linear_regression(oa, od).slope)
        if bd:
            bp_deltas.append(st.mean(bd))
            bp_slopes.append(st.linear_regression(ba, bd).slope)
    check("orthogonal_control delta", st.mean(orth_deltas), -0.0966, tol=0.001)
    check("band_probe delta", st.mean(bp_deltas), 0.1298, tol=0.001)
    check("orthogonal_control alpha slope", st.mean(orth_slopes), -0.288, tol=0.005)
    check("band_probe alpha slope", st.mean(bp_slopes), 0.550, tol=0.005)

    print("=== RQ3: alpha slopes (self ~0.589, alpha=0.5 self ~0.361) ===")
    alpha_rows = phase3_alpha_slope_rows(paired_rows)
    alpha_summary = phase3_alpha_summary(alpha_rows)
    check("self_alpha_slope_mean", alpha_summary["self_alpha_slope_mean"], 0.589, tol=0.005)
    check("other_alpha_slope_mean", alpha_summary["other_alpha_slope_mean"], 0.562, tol=0.005)

    self_at_alpha_05 = []
    for model_id in model_ids:
        by_alpha = phase3_category_summary(registry, model_id, run_scope)
        if 0.5 in by_alpha and "self" in by_alpha[0.5]:
            self_at_alpha_05.append(by_alpha[0.5]["self"])
    check("self @ alpha=0.5", st.mean(self_at_alpha_05), 0.361, tol=0.005)

    print("=== RQ1: peak-layer permutation baseline (models above the label-shuffle null) ===")
    perm_summary = {row["part"]: row for row in rq1_peak_permutation_summary(
        rq1_peak_permutation_rows(rq1_layer_part_rows(registry, model_ids, run_scope)))}
    for part, raw, corrected in ((1, 24, 9), (2, 25, 22), (3, 25, 25)):
        check(f"RQ1 part {part} n models above null, raw", perm_summary[part]["n_sig_raw"], raw)
        check(f"RQ1 part {part} n models above null, layer-corrected",
              perm_summary[part]["n_sig_layer_bonferroni"], corrected)
    perm_cells = rq1_peak_permutation_rows(rq1_layer_part_rows(registry, model_ids, run_scope))
    for part, at_floor, median_corrected in ((1, 9, 0.096), (2, 21, 0.034), (3, 25, 0.032)):
        part_cells = [r for r in perm_cells if r["part"] == part]
        check(f"RQ1 part {part} n models at the permutation floor (p = 1/1001)",
              sum(abs(r["perm_pvalue_raw"] - 1 / 1001) < 1e-9 for r in part_cells), at_floor, tol=0)
        check(f"RQ1 part {part} median layer-corrected p",
              st.median(r["perm_pvalue_layer_bonferroni"] for r in part_cells), median_corrected, tol=0.0006)
    cell = {(r["model_hf_id"], r["part"]): r for r in perm_cells}

    def part1_peaks_late(model: str) -> bool:
        position = {p: cell[(model, p)]["peak_layer"] / (cell[(model, p)]["n_layers"] - 1) for p in (1, 2, 3)}
        return sorted(position, key=lambda p: -position[p]).index(1) <= 1

    check("RQ1 Part 1 peaks latest or second-latest, all models",
          sum(part1_peaks_late(m) for m in model_ids), 23, tol=0)
    part1_passing = [m for m in model_ids if cell[(m, 1)]["sig_layer_bonferroni"]]
    check("RQ1 Part 1 passes corrected test: n models", len(part1_passing), 9, tol=0)
    check("RQ1 Part 1 peaks latest or second-latest, among those",
          sum(part1_peaks_late(m) for m in part1_passing), 8, tol=0)

    print("=== RQ3: pairwise self-unique vs other-unique check (884 cells, all 25 models) ===")
    # run_scope is deliberately not passed: the pairwise rows sit on later steer runs.
    pair_rows = phase3_pairwise_specificity_all_models(registry, model_ids)
    pooled = {row["subset"]: row for row in phase3_pairwise_specificity_pooled_rows(pair_rows)}
    all_pairs = pooled["all_pairs"]
    check("pairwise n cells", all_pairs["n_cells"], 884)
    check("pairwise n models", all_pairs["n_models"], 25)
    check("pairwise self_unique - noise", all_pairs["self_unique_minus_noise_mean"], 0.194, tol=0.002)
    check("pairwise other_unique - noise", all_pairs["other_unique_minus_noise_mean"], 0.181, tol=0.002)
    check("pairwise self-other gap", all_pairs["self_other_unique_gap_mean"], 0.0134, tol=0.001)
    check("pairwise gap CI low", all_pairs["self_other_unique_gap_ci_low"], -0.0153, tol=0.003)
    check("pairwise gap CI high", all_pairs["self_other_unique_gap_ci_high"], 0.0417, tol=0.003)
    check("pairwise specificity ratio", all_pairs["specificity_ratio"], 0.0452, tol=0.002)

    by_model: dict[str, tuple[list[float], list[float]]] = {}
    for row in registry.query(
            "SELECT r.model_hf_id AS model, s.human_band AS human, s.predicted_band AS machine"
            " FROM model_scores s JOIN experiment_runs r ON r.id = s.run_id"):
        humans, machines = by_model.setdefault(row["model"], ([], []))
        humans.append(row["human"])
        machines.append(row["machine"])
    agreement = [st.correlation(h, m) for h, m in by_model.values()]
    check("agreement n models", len(agreement), 25, tol=0)
    check("agreement median Pearson r", st.median(agreement), 0.41, tol=0.005)
    check("agreement n models r >= 0.6", sum(r >= 0.6 for r in agreement), 7, tol=0)
    check("agreement n models r >= 0.8", sum(r >= 0.8 for r in agreement), 0, tol=0)

    four = {"grammar", "vocabulary", "cohesion", "spelling_punctuation"}
    head_to_head = phase3_per_pair_equivalence_rows(registry, model_ids, run_scope=run_scope)
    testable = [r for r in head_to_head
                if r["self_criterion"] in four and r["other_criterion"] in four]
    check("head-to-head: testable ordered pairs among the 4 rubrics", len(testable), 10, tol=0)
    check("head-to-head: pairs where matching direction is detectably better",
          sum(r["model_level_gap_ci_low"] > 0 for r in testable), 0, tol=0)
    check("head-to-head: pairs where mismatched direction is detectably better",
          sum(r["model_level_gap_ci_high"] < 0 for r in testable), 1, tol=0)

    family = {r["hf_id"]: r["family"] for r in registry.query("SELECT hf_id, family FROM models")}
    model_gap = {r["model_hf_id"]: r["self_other_gap_mean"] for r in model_summary}
    leave_family_out = [st.mean(g for m, g in model_gap.items() if family[m] != f)
                        for f in set(family[m] for m in model_gap)]
    check("leave-one-family-out max gap", max(leave_family_out), 0.0083, tol=0.0002)

    agreeing = {m for m, (h, mc) in by_model.items() if st.correlation(h, mc) >= 0.6}
    subset_rows = [r for r in paired_rows if r["model_hf_id"] in agreeing]
    subset_summary = phase3_specificity_ratio_by_model_rows(subset_rows)
    subset = phase3_equivalence_summary(subset_rows, subset_summary)
    check("models with agreement r >= 0.6: n", subset["n_models"], 7, tol=0)
    check("models with agreement r >= 0.6: model-level gap", subset["model_level_gap_mean"], 0.0107, tol=0.001)
    check("models with agreement r >= 0.6: specificity ratio",
          subset["model_level_specificity_ratio_mean"], 0.0194, tol=0.002)
    check("models with agreement r >= 0.6: gap CI low", subset["model_level_gap_ci_low"], 0.0068, tol=0.0006)
    check("models with agreement r >= 0.6: gap CI high", subset["model_level_gap_ci_high"], 0.0151, tol=0.0006)

    print("=== RQ3: within-part matching advantage (18 rubric pairs) ===")
    within = phase3_within_part_matching_advantage_rows(registry, model_ids, run_scope=run_scope)
    check("within-part rubric pairs", len(within), 18, tol=0)
    check("within-part pairs with a detectable matching advantage",
          sum(r["advantage_detectable"] for r in within), 9, tol=0)
    check("within-part pairs with a detectably reversed advantage",
          sum(r["reversed_detectable"] for r in within), 1, tol=0)
    check("within-part largest matching advantage",
          max(r["matching_advantage_mean"] for r in within), 0.0301, tol=0.0005)
    check("within-part smallest matching advantage",
          min(r["matching_advantage_mean"] for r in within), -0.0154, tol=0.0005)
    detectable_sizes = sorted(r["relative_size"] for r in within if r["advantage_detectable"])
    check("within-part relative size, detectable pairs: median",
          st.median(detectable_sizes), 0.056, tol=0.003)
    check("within-part relative size, detectable pairs: max", max(detectable_sizes), 0.136, tol=0.003)

    registry.close()

    print()
    print(f"{'label':45} {'actual':>12} {'expected':>12}  ok")
    n_fail = 0
    for label, actual, expected, ok in CHECKS:
        if not ok:
            n_fail += 1
        print(f"{label:45} {actual:>12.4f} {expected:>12.4f}  {'PASS' if ok else 'FAIL'}")
    print()
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
