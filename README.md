# Inside the Scorer: code and data

Code and data for **"Inside the Scorer: Mechanistic Evidence on Rubric Control in LLM-Based Automated Essay Scoring from 25 Base Models"**
by [Chungche Huang](https://chungche.com), Po-Ya Angela Wang, and Shu-Kai Hsieh (National Taiwan University),
accepted at **[PACLIC 2026](https://paclic2026.github.io/)** (the 40th Pacific Asia Conference on Language, Information and Computation).

The paper probes and steers 25 open-weight base LLMs on human-scored English writing responses
from [BESTEP](https://bestep.tw) (培力英語能力檢定測驗, Taiwan's Ministry-of-Education English test)
to ask whether rubric criteria are represented inside a model and whether they can be controlled
one at a time. This repository lets you recompute every number in the paper from the
underlying database rows. It needs no GPU, no model weights, and no Ollama daemon.

## Quick start

Requires Python 3 and `numpy` (the bootstrap confidence intervals use `numpy.random.default_rng`); tested on
Python 3.12 and 3.14 with numpy 2.5. Nothing else. You need about 350 MB of free disk space, and the run takes
about 30 seconds.

```bash
git clone https://github.com/lopentu/inside-the-scorer-paclic2026
cd inside-the-scorer-paclic2026
python3 code/run_test.py
```

The first run unpacks `data/bestep_25models.db.gz` (33 MB) into `data/bestep_25models.db` (273 MB),
then recomputes the paper's numbers and prints a PASS/FAIL table. All 55 checks pass on the
database as shipped.

## What `run_test.py` checks

Each check recomputes a value straight from the database and compares it with the value in the paper.

| Paper result | What is recomputed |
|---|---|
| RQ1, peak-layer decodability against a permutation null | how many models' peak layer beats a label-shuffle null, raw and Bonferroni-corrected over layers (Part 1: 24 and 9 of 25; Part 2: 25 and 22; Part 3: 25 and 25) |
| RQ2, Table 4 | disattenuated criterion-pair similarity for all 12 comparable pairs, with cell counts |
| RQ3, headline | self versus other steering over 74,304 paired cells: cell-level gap 0.0085, model-level gap 0.0074 |
| RQ3, per-criterion effects | each criterion's own steering effect, 0.168 to 0.359 score points |
| RQ3, controls | orthogonal control (-0.097) and generic holistic direction (0.130) |
| RQ3, pairwise check | 884 pair cells: matching and mismatched residual directions clear the noise floor by 0.194 and 0.181, and their gap is 0.013 (95% CI -0.015 to 0.042) |
| RQ3, dose response | alpha slope 0.589 and the alpha = 0.5 effect of 0.361 score points |
| Limitations, agreement with human raters | each model's own expected band against the human band over the 75 essays: median Pearson r 0.41, 7 of 25 models at r >= 0.6, none at 0.8 |

To explore beyond these checks, import `analysis.py` directly:

```python
import sys
from pathlib import Path
sys.path.insert(0, "code")
from registry import Registry
from analysis import paper_model_ids, build_run_audit, phase1_layer_summary

registry = Registry(Path("data/bestep_25models.db"))
model_ids = paper_model_ids()
run_scope, _ = build_run_audit(registry, model_ids)
print(phase1_layer_summary(registry, model_ids[0], run_scope=run_scope))
```

## Contents

```
code/
  analysis.py    every analysis function behind a number in the paper (RQ1 decodability and its
                 permutation baseline, RQ2 criterion-pair geometry, RQ3 steering with the
                 orthogonal and generic-holistic controls, and the pairwise check)
  registry.py    minimal read-only SQLite wrapper that analysis.py depends on
  context.py     the BESTEP grading prompt template and rubric text, identical for every model
                 (probes and steering read the activation at this prompt's final token)
  mutation.py    the 12 rubric-quoting specs used to generate the degraded essays
  run_test.py    reproduces the paper's numbers and checks them against the published values
data/
  bestep_25models.db.gz   the database for the 25 paper models, gzip-compressed
CITATION.cff
```

### The database

`bestep_25models.db` holds exactly the rows the analysis reads for the 25 models in
`analysis.paper_model_ids()`, using the run selection that produced the published numbers (one
completed run per model per phase). It contains no essay text, only candidate identifiers, human band
scores, and model outputs.

| Table | Rows | What it holds |
|---|---:|---|
| `models` | 25 | model catalog (parameters, layers, family) |
| `experiment_runs` | 102 | the selected completed runs, plus the 25 later steering runs that hold the pairwise rows |
| `probes` | 11,610 | RQ1: per-layer R² and permutation p-values |
| `contrastive_geometry` | 51,084 | RQ2: per-layer criterion-direction cosine similarities |
| `steering_metrics` | 1,136,688 | RQ3: steering results (668,736 self/other/orthogonal/band-probe rows across 4 alphas, plus 467,952 pairwise and noise-floor rows) |
| `model_scores` | 1,875 | holistic band-score predictions against human scores |
| `artifacts` | 25 | pointers to the activation files (optional) |

## What is and is not included

Included: the full analysis chain from stored results to every reported number.

Not included, because they need infrastructure this repository deliberately leaves out:

- **Generating degraded essays.** `mutation.py` shows exactly how the 12 degradation specs were written, but
  producing new ones needs a live Ollama daemon running `llama3.3:70b`.
- **Extracting activations and steering.** This needs a GPU, downloaded model weights, and TransformerLens.
  The database stores what those steps produced.
- **The degraded essay pairs.** The 104 degraded essays (20 for Part 1, 36 for Part 2, 48 for Part 3) and
  the originals they were made from are exam responses, so they are not public yet. They are available for
  research purposes on request: email ovoid.amylase_8i@icloud.com. Each of the 104 was confirmed by two
  trained BESTEP graders (see Appendix A of the paper).

If a number in the paper is not covered by `run_test.py`, the computation follows the pattern of the
functions in `analysis.py`: query the relevant table, gated by the same 25-model `run_scope`.

## License

The code and the database are released under the
[PolyForm Noncommercial License 1.0.0](LICENSE): you may use, copy, and modify them for research, teaching,
and other noncommercial purposes, and commercial use is not permitted. This is a source-available license,
not an OSI open-source license. Copyright (c) 2026 [Chungche Huang](https://chungche.com). The license does not cover the exam
essays, which are not included (see above). For commercial use, contact the author.

## Citation

Please cite the paper. See `CITATION.cff`; a BibTeX entry with page numbers will be added when the
PACLIC 2026 proceedings are published.
