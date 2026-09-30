"""Phase D: expert judgment study on the interpretability of the linguistic outputs
(Sections 6.4 and 7.4 of the guide).

Domain experts are shown explanations of the same predictions in three
formats, namely a feature level SHAP bar plot, a plain concept level bar plot
without uncertainty, and the fuzzy linguistic explanation of PFCA, and rate
confidence calibration, ease of understanding and actionability on Likert
scales from 1 to 7. They also answer three factual questions per prediction
whose correct answers depend on the reliability information: which of the
concepts shown has the least reliable sign (lowest sign confidence among the
retained concepts), whether the model classes disagree on the most important
concept (disagreement index above the threshold), and whether the contribution
of the most important concept is reliably positive, reliably negative or
uncertain (sign confidence at least 0.9 together with the sign of the median).

The script has two subcommands.

generate
    Fits PFCAExplainer on a problem (the questionnaire of Phase C, or a
    synthetic problem with --synthetic), explains a pool of candidate
    instances, chooses the presented instances so that the answer key is as
    balanced as possible, renders the three formats of every presented
    prediction and writes the study materials. The formats are hidden behind
    the codes F1, F2 and F3 (a seeded permutation), so that a file name does
    not reveal the format; the mapping is written to a private folder together
    with the answer key. The design is within subject with counterbalanced
    order: every expert sees every prediction in every format, the formats are
    presented in blocks whose order follows a balanced Latin square (Williams
    design, six sequences for three formats), and the order of the predictions
    inside every block is a seeded permutation.

analyze
    Reads the filled response sheets, merges the private mapping and the answer
    key, and writes descriptive statistics per format, the Friedman test with
    the Nemenyi post hoc procedure across the three formats, Wilcoxon signed
    rank tests with Holm correction against the fuzzy format, an ordinal model
    per rating with the format as predictor, and a binomial mixed model of
    correctness with the format as fixed effect and the expert as random
    intercept. With --simulate the blank response sheet of the run is filled
    with random plausible responses first, so that the analysis path can be tested;
    the outputs of a simulated analysis are marked as such and carry no
    evidence.

Files written by generate (inside results/<name>/)
--------------------------------------------------
config.yaml, environment.json             frozen configuration and environment (experiments/common.prepare_run)
materials/instance_<P>_<F>.png            explanation of prediction P in the format with code F
materials/instance_<P>_<F>.txt            text of the linguistic explanation (only for the fuzzy format)
materials/concept_dictionary.txt          concepts and their member inputs, given to every participant
materials/instructions.txt                instructions for participants
materials/sheets/response_sheet_<E>.csv   response sheet of expert E (rows in presentation order, ratings and answers blank)
response_sheet.csv                        all response sheets stacked (the file analyze expects back, filled in)
private/format_mapping.csv                format code to format name (do not give to participants)
private/answer_key.csv                    correct answers of the three questions per prediction with the supporting quantities
private/instance_mapping.csv              presentation id to candidate index and model output
private/instances.csv                     input values of the presented instances
private/design.csv                        format sequence of every expert
private/explanation.json                  the PFCA explanation of the candidate pool (PFCAExplanation.save)
private/seeds.json                        every seed used

Files written by analyze (inside results/<name>/analysis/ or analysis_simulated/)
---------------------------------------------------------------------------------
scored_responses.csv                      the merged and scored long table
descriptives_ratings.csv                  mean, sd, median and n of every rating per format
descriptives_accuracy.csv                 accuracy per question and format
friedman.csv, nemenyi_pairs.csv           Friedman tests and Nemenyi pairwise decisions per outcome
wilcoxon_vs_fuzzy.csv                     Wilcoxon signed rank tests against the fuzzy format, Holm corrected
ordinal_models.csv, ordinal_models.txt    proportional odds models per rating (reference format: fuzzy)
accuracy_mixed_model.csv, accuracy_mixed_model.txt   binomial mixed model of correctness
ratings_by_format.png                     expert means per format
summary.txt                               short text summary

Usage
-----
python experiments/run_phase_d.py generate --synthetic --n-instances 3 --n-experts 6 --results results --name phase_d_smoke --n-resamples 3
python experiments/run_phase_d.py analyze --results results --name phase_d_smoke --simulate
python experiments/run_phase_d.py generate --config experiments/configs/phase_d.yaml --results results --name phase_d
python experiments/run_phase_d.py generate --csv data/demo_questionnaire.csv --spec data/demo_questionnaire_spec.json --results results --name phase_d_demo
python experiments/run_phase_d.py analyze --results results --name phase_d --responses results/phase_d/responses.csv
python experiments/run_phase_d.py analyze --results results --name phase_d --responses returned/expert_E01.csv returned/expert_E02.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, model_classes_from_names, prepare_run, tune_gradient_boosting  # noqa: E402

import pfca.plotting as plotting  # noqa: E402
from pfca.evaluation import statistics as S  # noqa: E402
from pfca.evaluation.datasets import load_questionnaire, stratified_splits  # noqa: E402
from pfca.evaluation.synthetic import make_synthetic  # noqa: E402
from pfca.explainer import PFCAExplainer  # noqa: E402
from pfca.plotting import MUTED, PALETTE, TEXT, plot_linguistic_bars, plot_shap_bars, style_axes  # noqa: E402

FORMAT_ORDER = ["shap", "concept", "fuzzy"]
FORMAT_DESCRIPTION = {
    "shap": "feature level SHAP bar plot",
    "concept": "plain concept level bar plot without uncertainty",
    "fuzzy": "fuzzy linguistic explanation (bars with support whiskers and labels, plus text)",
}
RATINGS = ["rating_confidence_calibration", "rating_ease", "rating_actionability"]
QUESTIONS = ["q1", "q2", "q3"]
SHEET_COLUMNS = ["expert_id", "position", "instance_id", "format_code"] + RATINGS + [f"answer_{q}" for q in QUESTIONS]
Q3_ANSWERS = ("positive", "negative", "uncertain")


# ----------------------------------------------------------------------
# helpers shared by both subcommands
# ----------------------------------------------------------------------


def resolve_path(path: str) -> Path:
    """Resolve a data path relative to the working directory, then to the repository root."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    alt = ROOT / p
    return alt if alt.exists() else p


def concept_code(k: int) -> str:
    return f"C{int(k) + 1}"


def williams_square(k: int) -> list[list[int]]:
    """Rows of a balanced Latin square (Williams design) for k treatments.

    Every treatment appears once in every position of every row, and every
    ordered pair of adjacent treatments appears equally often over the rows,
    which balances first order carryover. For odd k two squares are needed
    (the second is the reverse of the first), giving 2k rows.
    """
    base = [0]
    lo, hi = 1, k - 1
    while len(base) < k:
        base.append(lo)
        lo += 1
        if len(base) < k:
            base.append(hi)
            hi -= 1
    rows = [[(b + i) % k for b in base] for i in range(k)]
    if k % 2 == 1:
        rows = rows + [list(reversed(r)) for r in rows]
    return rows


# ----------------------------------------------------------------------
# generate: problem, explanation, answer key
# ----------------------------------------------------------------------


def load_problem(cfg: dict, seed: int) -> dict:
    """Training, tuning and candidate data of the problem, with feature names and a priori groups."""
    n_candidates = int(cfg["n_candidates"])
    if cfg.get("use_synthetic"):
        p = cfg["synthetic"]
        problem = make_synthetic(p["family"], n_samples=int(p["n_samples"]), n_features=int(p["n_features"]), rho=float(p["rho"]), random_state=seed)
        n = problem.X.shape[0]
        X_tune, y_tune = problem.sample(max(50, int(float(p.get("tune_fraction", 0.34)) * n)), random_state=seed + 1)
        X_cand, _ = problem.sample(n_candidates, random_state=seed + 2)
        return {
            "name": f"synthetic_{p['family']}",
            "task": "regression",
            "X_train": problem.X,
            "y_train": problem.y,
            "X_tune": X_tune,
            "y_tune": y_tune,
            "X_cand": X_cand,
            "feature_names": list(problem.feature_names),
            "groups": [list(map(int, b)) for b in problem.blocks],
            "group_names": [f"factor{k + 1}" for k in range(problem.n_factors)],
            "meta": {"family": p["family"], "n_samples": int(n), "n_features": int(problem.n_features), "rho": float(problem.rho), "noise_sd": float(problem.noise_sd)},
        }
    csv_path, spec_path = resolve_path(cfg["csv_path"]), resolve_path(cfg["spec_path"])
    if not csv_path.exists() or not spec_path.exists():
        raise SystemExit(f"Questionnaire files not found: {csv_path} / {spec_path}. Use --synthetic, or see data/README.md and experiments/make_demo_questionnaire.py.")
    ds = load_questionnaire(str(csv_path), str(spec_path))
    _, idx_train, idx_tune, idx_test = next(stratified_splits(ds.y, ds.task, n_repeats=1, random_state=seed))
    idx_cand = idx_test[: min(n_candidates, idx_test.size)]
    return {
        "name": ds.name,
        "task": ds.task,
        "X_train": ds.X[idx_train],
        "y_train": ds.y[idx_train],
        "X_tune": ds.X[idx_tune],
        "y_tune": ds.y[idx_tune],
        "X_cand": ds.X[idx_cand],
        "feature_names": list(ds.feature_names),
        "groups": ds.groups,
        "group_names": list(ds.groups.keys()) if ds.groups else [],
        "meta": {"csv_path": str(csv_path), "spec_path": str(spec_path), "n_rows": int(ds.X.shape[0]), "n_train": int(idx_train.size), "n_tune": int(idx_tune.size), "n_test": int(idx_test.size)},
    }


def fit_explanation(prob: dict, cfg: dict, seed: int):
    """Tune the reference model, fit the PFCA explainer and explain the candidate instances."""
    task = prob["task"]
    ref, ref_params, ref_score = tune_gradient_boosting(prob["X_train"], prob["y_train"], prob["X_tune"], prob["y_tune"], task, seed, grid=cfg.get("tuning_grid"))
    classes = model_classes_from_names(cfg["model_classes"], task, seed)
    classes[0] = ("gbm", ref.__class__(**ref.get_params()))
    d = len(prob["feature_names"])
    pfca_kwargs = dict(cfg["pfca"])
    pfca_kwargs["n_concepts_grid"] = tuple(k for k in pfca_kwargs.get("n_concepts_grid", (2, 3, 4, 5, 6)) if k < d)
    explainer = PFCAExplainer(
        model_classes=classes,
        n_resamples=int(cfg["n_resamples"]),
        background_size=cfg.get("background_size", 100),
        explainer=cfg.get("explainer", "auto"),
        apriori_groups=prob["groups"] if prob["groups"] else None,
        partition_source="both" if prob["groups"] else "data",
        task=task,
        n_jobs=int(cfg.get("n_jobs", 1)),
        random_state=seed,
        **pfca_kwargs,
    )
    t = time.time()
    explainer.fit(prob["X_train"], prob["y_train"], prob["feature_names"])
    expl = explainer.explain(prob["X_cand"])
    info = {"reference_params": ref_params, "reference_tuning_score": float(ref_score), "fit_seconds": time.time() - t, "model_classes": [n for n, _ in classes]}
    return explainer, expl, info


def instance_key(expl, i: int, reliable: float) -> dict:
    """Correct answers of the three factual questions for explained instance i.

    q1: retained concept(s) with the lowest sign confidence (ties joined by "|").
    q2: whether the disagreement index of the most important retained concept
        (largest absolute centroid) exceeds the threshold of the explanation.
    q3: positive or negative when the sign confidence of the most important
        concept is at least ``reliable`` (sign of the median), uncertain otherwise.
    """
    retained = [int(k) for k in expl.retained]
    if not retained:
        raise ValueError("The explanation retains no concept; lower the alpha grid or increase the number of resamples.")
    cen = expl.concept_centroids[i]
    med = expl.concept_medians[i]
    sc = expl.sign_confidence[i]
    dis = expl.disagreement[i]
    top = max(retained, key=lambda k: abs(cen[k]))
    min_sc = min(float(sc[k]) for k in retained)
    q1 = [k for k in retained if float(sc[k]) <= min_sc + 1e-9]
    q2 = "yes" if float(dis[top]) > float(expl.disagreement_threshold) else "no"
    sign = float(np.sign(med[top])) or float(np.sign(cen[top]))
    if float(sc[top]) >= reliable and sign != 0.0:
        q3 = "positive" if sign > 0 else "negative"
    else:
        q3 = "uncertain"
    return {
        "candidate_index": int(i),
        "model_output": float(expl.reference_output[i]),
        "retained_concepts": "|".join(concept_code(k) for k in retained),
        "n_retained": len(retained),
        "top_concept": concept_code(top),
        "top_centroid": float(cen[top]),
        "top_median": float(med[top]),
        "top_sign_confidence": float(sc[top]),
        "top_disagreement": float(dis[top]),
        "q1_key": "|".join(concept_code(k) for k in q1),
        "q1_tied": len(q1) > 1,
        "q1_sign_confidence": min_sc,
        "q2_key": q2,
        "q3_key": q3,
        "disagreement_threshold": float(expl.disagreement_threshold),
        "reliable_sign_confidence": float(reliable),
    }


def choose_instances(keys: list[dict], n: int, rng: np.random.Generator) -> list[int]:
    """Seeded choice of n candidates, round robin over the (q3, q2) answer combinations so that the key is balanced.

    Inside every combination candidates whose first question has a unique
    answer (no tie in the lowest sign confidence) are taken first.
    """
    strata: dict[tuple, list[int]] = {}
    for j, key in enumerate(keys):
        strata.setdefault((key["q3_key"], key["q2_key"]), []).append(j)
    cats = list(strata)
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(strata[c])
        strata[c].sort(key=lambda j: bool(keys[j]["q1_tied"]), reverse=True)  # stable: untied candidates end up last and are popped first
    chosen: list[int] = []
    while len(chosen) < n and any(strata.values()):
        for c in cats:
            if strata[c] and len(chosen) < n:
                chosen.append(strata[c].pop())
    return sorted(chosen)


# ----------------------------------------------------------------------
# generate: materials
# ----------------------------------------------------------------------


def plot_plain_concept_bars(explanation, instance: int, retained_only: bool = True, ax=None):
    """Concept level bar plot without uncertainty: bars of the concept centroids only, no whiskers, no labels."""
    plt = plotting._plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.2))
    names = explanation.concept_names
    idx = list(explanation.retained) if retained_only else list(range(explanation.n_concepts))
    idx = sorted(idx, key=lambda k: abs(explanation.concept_centroids[instance, k]))
    cen = np.array([explanation.concept_centroids[instance, k] for k in idx])
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in cen]
    y = np.arange(len(idx))
    ax.barh(y, cen, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in idx])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("attribution to model output")
    style_axes(ax)
    return ax


def plot_fuzzy_format(explanation, instance: int):
    """Linguistic bars of pfca.plotting with every label placed to the right of the whisker.

    plot_linguistic_bars writes the label of a negative bar to the left of the
    support, where it can run into the concept name on the axis; for the study
    material every label is moved to the right end of the support instead.
    """
    ax = plot_linguistic_bars(explanation, instance)
    idx = sorted(list(explanation.retained), key=lambda k: abs(explanation.concept_centroids[instance, k]))
    for yi, k in enumerate(idx):
        if explanation.concept_centroids[instance, k] >= 0:
            continue
        for t in ax.texts:
            if abs(t.get_position()[1] - yi) < 1e-9 and t.get_ha() == "right":
                t.set_position((explanation.quantiles[instance, k, 4], yi))
                t.set_ha("left")
                t.set_text("  " + t.get_text().strip())
    return ax


def render_materials(expl, chosen: list[int], ids: list[str], feature_names: list[str], codes: dict, mat_dir: Path, shap_top: int) -> list[str]:
    """Write the three formats of every presented prediction under the format codes."""
    import matplotlib.pyplot as plt

    written = []
    for i, pid in zip(chosen, ids):
        title = f"Prediction {pid}: model output {expl.reference_output[i]:.4g}"
        ax = plot_shap_bars(expl.attribution.values[0, 0, i], feature_names, top=shap_top, title=title)
        path = mat_dir / f"instance_{pid}_{codes['shap']}.png"
        ax.figure.savefig(path, bbox_inches="tight")
        plt.close(ax.figure)
        written.append(path.name)
        ax = plot_plain_concept_bars(expl, i)
        ax.set_title(title, loc="left")
        path = mat_dir / f"instance_{pid}_{codes['concept']}.png"
        ax.figure.savefig(path, bbox_inches="tight")
        plt.close(ax.figure)
        written.append(path.name)
        ax = plot_fuzzy_format(expl, i)
        ax.set_title(title, loc="left")
        path = mat_dir / f"instance_{pid}_{codes['fuzzy']}.png"
        ax.figure.savefig(path, bbox_inches="tight")
        plt.close(ax.figure)
        written.append(path.name)
        lines = [ln for ln in expl.describe(i, retained_only=True).splitlines() if not ln.startswith("Configuration:")]
        lines[0] = title
        path = mat_dir / f"instance_{pid}_{codes['fuzzy']}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf8")
        written.append(path.name)
    return written


def concept_dictionary_text(expl, feature_names: list[str], min_degree: float = 0.05) -> str:
    """Concepts with their member inputs and membership degrees, for the participants."""
    U = expl.partition.U
    lines = [
        "Concept dictionary",
        "",
        "The inputs of the model are grouped into concepts. Every input belongs to each concept to a degree between 0 and 1,",
        "and the degrees of one input sum to one across the concepts. The list below gives, for every concept, the inputs",
        f"that belong to it with a degree of at least {min_degree:.2f}, from the strongest member to the weakest.",
        "Concepts marked as shown are the ones that appear in the concept level explanations.",
        "",
    ]
    retained = set(int(k) for k in expl.retained)
    for k in range(expl.n_concepts):
        order = np.argsort(-U[:, k])
        members = [f"{feature_names[j]} ({U[j, k]:.2f})" for j in order if U[j, k] >= min_degree]
        shown = "shown" if k in retained else "not shown"
        lines.append(f"{concept_code(k)} ({shown}): " + ", ".join(members))
    lines.append("")
    lines.append("Full name of every concept as it appears in the figures:")
    for k, name in enumerate(expl.concept_names):
        lines.append(f"  {name}")
    return "\n".join(lines) + "\n"


def instructions_text(n_instances: int, n_positions: int, likert: tuple[int, int], label_names: list[str]) -> str:
    lo, hi = likert
    labels = ", ".join(label_names)
    return f"""Instructions for participants

Thank you for taking part in this study on the presentation of explanations of model predictions.

1. What you will see

A predictive model was fitted to a dataset in which every case is described by a number of inputs. For {n_instances} predictions of this model you will see explanations, that is, descriptions of which inputs, or which groups of inputs, pushed the prediction up or down and by how much. The same prediction is explained in more than one way, so you will see every prediction several times. Every explanation is one image file (PNG); some explanations come with a text file (TXT) of the same name. The files are in the folder "materials" and are named instance_<prediction>_<format>.png, for example instance_P02_F3.png. When a text file with the same name exists, read the image and the text together. The heading of every explanation shows the identifier of the prediction and the output of the model for that case.

The inputs of the model are grouped into concepts named C1, C2 and so on. The file concept_dictionary.txt lists the inputs that belong to every concept and the degree to which they belong. Some explanations refer to concepts and others to single inputs, so keep the dictionary at hand while you work.

2. Your response sheet

Your response sheet (materials/sheets/response_sheet_<your id>.csv) has {n_positions} rows, one per item, in the order in which you should work through them. The columns instance_id and format_code name the files to open for the item. Work through the rows in the given order, complete one row before opening the next item, and do not return to earlier rows. Fill in the six empty cells of every row and leave the other cells unchanged. Save the file in CSV format under the same name.

3. Ratings

Rate every explanation on three statements with a whole number from {lo} to {hi}, where {lo} means strongly disagree and {hi} means strongly agree.

rating_confidence_calibration   The explanation makes clear how much each stated contribution can be trusted.
rating_ease                     The explanation is easy to understand.
rating_actionability            The explanation helps me decide what to examine or do next for this case.

4. Questions

Answer three questions about every explanation. Base your answers only on the material of the current item and the concept dictionary. Answer every question with one of the allowed answers, even when you have to guess. The most important concept is the concept whose contribution is largest in absolute value among the concepts shown; when the explanation refers to single inputs, use the dictionary to identify the concept to which the most important inputs belong.

answer_q1   Which of the concepts shown has the least reliable sign, that is, for which concept is it least certain whether its contribution is positive or negative? Write the concept code, for example C2.
answer_q2   Do the different types of model disagree about the contribution of the most important concept? Write yes or no.
answer_q3   Is the contribution of the most important concept reliably positive, reliably negative, or uncertain? Write positive, negative or uncertain.

5. Terms that may appear in the materials

contribution, attribution     The amount by which an input or a concept moves the output of the model for this case; positive values push the output up, negative values push it down.
centroid                      The single number that summarizes a contribution when the contribution is given as a range.
support, core                 Ranges of plausible values of a contribution; the support is the wider range and the core the narrower one.
sign confidence               The proportion of repeated estimates of a contribution that have the same sign as the typical estimate; a value of 1.00 means that every estimate had the same sign.
disagreement                  The extent to which different types of model attribute a different contribution to a concept, between 0 and 1. A note that the model classes disagree means that this value is above a fixed threshold.
selection membership          The proportion of acceptable explanation settings in which a concept is kept, between 0 and 1.
compatibility                 The degree, between 0 and 1, to which a contribution fits a verbal label. The labels are: {labels}.
configuration                 Technical settings of the explanation method; they can be ignored.

6. Time

Take the time you need. When you have finished, return your response sheet and, if you wish, remarks in a separate text file. Do not discuss the material with other participants before all sheets have been returned.
"""


# ----------------------------------------------------------------------
# generate: design and response sheets
# ----------------------------------------------------------------------


def build_design(n_experts: int, ids: list[str], codes: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Response sheet (blank) and private design table.

    Formats are presented in blocks. The block order of every expert is a row
    of the balanced Latin square; the rows are assigned in complete sets of
    six, each set in a seeded random order, so that any multiple of six
    experts gives a fully balanced design. The order of the predictions inside
    every block is an independent seeded permutation.
    """
    rows_ls = williams_square(len(FORMAT_ORDER))
    seqs: list[list[int]] = []
    for _ in range(math.ceil(n_experts / len(rows_ls))):
        order = rng.permutation(len(rows_ls))
        seqs += [rows_ls[o] for o in order]
    seqs = seqs[:n_experts]
    sheet_rows, design_rows = [], []
    for e, seq in enumerate(seqs):
        eid = f"E{e + 1:02d}"
        names = [FORMAT_ORDER[f] for f in seq]
        design_rows.append({"expert_id": eid, "latin_square_row": int(rows_ls.index(seq)), "format_sequence": "|".join(names), "code_sequence": "|".join(codes[n] for n in names)})
        position = 0
        for fmt in names:
            perm = rng.permutation(len(ids))
            for j in perm:
                position += 1
                row = {"expert_id": eid, "position": position, "instance_id": ids[int(j)], "format_code": codes[fmt]}
                row.update({c: "" for c in RATINGS})
                row.update({f"answer_{q}": "" for q in QUESTIONS})
                sheet_rows.append(row)
    return pd.DataFrame(sheet_rows, columns=SHEET_COLUMNS), pd.DataFrame(design_rows)


def run_generate(args) -> Path:
    overrides = {
        "n_instances": args.n_instances,
        "n_experts": args.n_experts,
        "n_candidates": args.n_candidates,
        "n_resamples": args.n_resamples,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "csv_path": args.csv,
        "spec_path": args.spec,
    }
    cfg = load_config(args.config, overrides)
    cfg["use_synthetic"] = bool(args.synthetic)
    seed = int(cfg.get("seed", 0))
    n_instances, n_experts = int(cfg["n_instances"]), int(cfg["n_experts"])
    cfg["n_candidates"] = max(int(cfg.get("n_candidates", 60)), n_instances)
    if n_experts % len(williams_square(3)) != 0:
        print(f"[phase D] warning: {n_experts} experts is not a multiple of six, the format order is balanced only within complete sets of six experts")
    out_dir = prepare_run(args.results, args.name, cfg)
    mat_dir = out_dir / "materials"
    sheet_dir = mat_dir / "sheets"
    priv_dir = out_dir / "private"
    for d in (mat_dir, sheet_dir, priv_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[phase D] generate: seed {seed}, {n_instances} predictions, {n_experts} experts, results in {out_dir}")

    prob = load_problem(cfg, seed)
    print(f"[phase D] problem {prob['name']}: task {prob['task']}, {prob['X_train'].shape[0]} training rows, {len(prob['feature_names'])} features, {prob['X_cand'].shape[0]} candidate instances", flush=True)
    explainer, expl, info = fit_explanation(prob, cfg, seed)
    cfgk = expl.configuration
    print(f"[phase D] knee explanation: partition {cfgk.partition}, K={cfgk.n_concepts}, sparsity={cfgk.sparsity}, alpha={cfgk.alpha:.2f}, retained {[concept_code(k) for k in expl.retained]} ({info['fit_seconds']:.1f}s)", flush=True)
    if expl.retained.size < 2:
        print("[phase D] warning: fewer than two concepts are retained, question 1 has a single possible answer")

    # answer key of every candidate, choice of the presented instances
    reliable = float(cfg.get("reliable_sign_confidence", 0.9))
    keys = [instance_key(expl, i, reliable) for i in range(expl.n_explain)]
    rng = np.random.default_rng(seed + 100)
    chosen = choose_instances(keys, n_instances, rng)
    ids = [f"P{j + 1:02d}" for j in range(len(chosen))]
    code_perm = rng.permutation(len(FORMAT_ORDER))
    codes = {fmt: f"F{int(code_perm[j]) + 1}" for j, fmt in enumerate(FORMAT_ORDER)}

    # materials
    written = render_materials(expl, chosen, ids, prob["feature_names"], codes, mat_dir, int(cfg.get("shap_top_features", 15)))
    (mat_dir / "concept_dictionary.txt").write_text(concept_dictionary_text(expl, prob["feature_names"]), encoding="utf8")
    likert = (int(cfg.get("likert_min", 1)), int(cfg.get("likert_max", 7)))
    sheet, design = build_design(n_experts, ids, codes, rng)
    n_positions = int(sheet.groupby("expert_id")["position"].max().iloc[0])
    (mat_dir / "instructions.txt").write_text(instructions_text(len(ids), n_positions, likert, expl.label_set.names), encoding="utf8")
    sheet.to_csv(out_dir / "response_sheet.csv", index=False)
    for eid, part in sheet.groupby("expert_id", sort=True):
        part.to_csv(sheet_dir / f"response_sheet_{eid}.csv", index=False)

    # private files
    pd.DataFrame([{"format_code": codes[f], "format": f, "description": FORMAT_DESCRIPTION[f]} for f in FORMAT_ORDER]).sort_values("format_code").to_csv(priv_dir / "format_mapping.csv", index=False)
    key_rows = []
    for pid, i in zip(ids, chosen):
        row = {"instance_id": pid}
        row.update(keys[i])
        key_rows.append(row)
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(priv_dir / "answer_key.csv", index=False)
    key_df[["instance_id", "candidate_index", "model_output"]].to_csv(priv_dir / "instance_mapping.csv", index=False)
    inst = pd.DataFrame(prob["X_cand"][chosen], columns=prob["feature_names"])
    inst.insert(0, "instance_id", ids)
    inst.insert(1, "candidate_index", chosen)
    inst.to_csv(priv_dir / "instances.csv", index=False)
    design.to_csv(priv_dir / "design.csv", index=False)
    expl.save(str(priv_dir / "explanation.json"))
    seeds = {
        "seed": seed,
        "problem_seed": seed,
        "design_seed": seed + 100,
        "pool_seeds": expl.attribution.seeds.tolist() if expl.attribution.seeds is not None else None,
        "format_codes": codes,
        "chosen_candidates": [int(c) for c in chosen],
        "problem": prob["meta"],
        "reference_params": info["reference_params"],
        "reference_tuning_score": info["reference_tuning_score"],
        "model_classes": info["model_classes"],
        "knee_configuration": cfgk.as_dict(),
        "retained_concepts": [concept_code(k) for k in expl.retained],
        "concept_names": expl.concept_names,
    }
    with open(priv_dir / "seeds.json", "w", encoding="utf8") as fh:
        json.dump(seeds, fh, indent=2)
    print(f"[phase D] wrote {len(written)} material files, {len(sheet)} response rows for {n_experts} experts, key for {len(ids)} predictions")
    print("[phase D] answer key summary: " + ", ".join(f"{pid}: q1 {k['q1_key']}, q2 {k['q2_key']}, q3 {k['q3_key']}" for pid, k in zip(ids, key_rows)))
    print("[phase D] generate finished")
    return out_dir


# ----------------------------------------------------------------------
# analyze: reading, scoring, simulation
# ----------------------------------------------------------------------


def read_responses(paths: list[Path]) -> pd.DataFrame:
    """Read one or more response sheets; repeated header lines from concatenated files are dropped.

    Files saved by spreadsheet programs may carry a byte order mark or use a
    semicolon as separator; both are accepted.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=";")
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in SHEET_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"Response sheet {p} lacks the columns {missing}")
        df = df[SHEET_COLUMNS]
        df = df[df["expert_id"] != "expert_id"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in SHEET_COLUMNS:
        df[c] = df[c].astype(str).str.strip()
    df = df[df["expert_id"] != ""]
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    return df.reset_index(drop=True)


def normalize_answer(question: str, answer) -> str:
    a = str(answer).strip().lower()
    if a in ("", "nan", "none"):
        return ""
    if question == "q1":
        m = re.search(r"c\s*0*(\d+)", a) or re.fullmatch(r"\s*0*(\d+)\s*", a)
        return f"C{int(m.group(1))}" if m else a.upper()
    if question == "q2":
        if a in ("y", "yes", "true", "1"):
            return "yes"
        if a in ("n", "no", "false", "0"):
            return "no"
        return a
    if a.startswith("pos") or a == "+":
        return "positive"
    if a.startswith("neg") or a == "-":
        return "negative"
    if a.startswith("unc") or a.startswith("uns"):
        return "uncertain"
    return a


def score_responses(df: pd.DataFrame, mapping: pd.DataFrame, key: pd.DataFrame, likert: tuple[int, int]) -> pd.DataFrame:
    """Merge the format mapping and the answer key, coerce the ratings and score the answers."""
    code_to_format = dict(zip(mapping["format_code"].astype(str), mapping["format"].astype(str)))
    unknown = sorted(set(df["format_code"]) - set(code_to_format))
    if unknown:
        raise SystemExit(f"Unknown format codes in the responses: {unknown}")
    df = df.copy()
    df["format"] = df["format_code"].map(code_to_format)
    key = key.copy()
    key["instance_id"] = key["instance_id"].astype(str)
    unknown = sorted(set(df["instance_id"]) - set(key["instance_id"]))
    if unknown:
        raise SystemExit(f"Unknown instance ids in the responses: {unknown}")
    df = df.merge(key[["instance_id", "q1_key", "q2_key", "q3_key", "top_concept", "retained_concepts"]], on="instance_id", how="left")
    lo, hi = likert
    for r in RATINGS:
        v = pd.to_numeric(df[r], errors="coerce")
        bad = v.notna() & ((v < lo) | (v > hi) | (v != np.round(v)))
        if bad.any():
            print(f"[phase D] warning: {int(bad.sum())} values of {r} outside {lo} to {hi} or not whole numbers are treated as missing")
        df[r] = v.where(~bad)
    for q in QUESTIONS:
        norm = df[f"answer_{q}"].map(lambda a, q=q: normalize_answer(q, a))
        df[f"answer_{q}_normalized"] = norm
        correct = []
        for a, k in zip(norm, df[f"{q}_key"].astype(str)):
            if a == "":
                correct.append(np.nan)
            else:
                correct.append(1.0 if a in k.split("|") else 0.0)
        df[f"correct_{q}"] = correct
    df["n_correct"] = df[[f"correct_{q}" for q in QUESTIONS]].sum(axis=1, min_count=1)
    df["accuracy"] = df[[f"correct_{q}" for q in QUESTIONS]].mean(axis=1)
    return df


def simulate_responses(sheet: pd.DataFrame, mapping: pd.DataFrame, key: pd.DataFrame, sim: dict, likert: tuple[int, int], rng: np.random.Generator) -> pd.DataFrame:
    """Fill a blank response sheet with random plausible responses (for testing the analysis path only)."""
    code_to_format = dict(zip(mapping["format_code"].astype(str), mapping["format"].astype(str)))
    keyed = key.set_index(key["instance_id"].astype(str))
    lo, hi = likert
    experts = sorted(sheet["expert_id"].unique())
    intercept = {e: rng.normal(0.0, float(sim.get("expert_sd", 0.5))) for e in experts}
    df = sheet.copy()
    for idx, row in df.iterrows():
        fmt = code_to_format[str(row["format_code"])]
        k = keyed.loc[str(row["instance_id"])]
        for r in RATINGS:
            mean = float(sim["rating_means"][r][fmt]) + intercept[row["expert_id"]]
            value = int(np.clip(np.round(mean + rng.normal(0.0, float(sim.get("rating_sd", 1.0)))), lo, hi))
            df.at[idx, r] = str(value)
        acc = float(sim["accuracy"][fmt])
        options_q1 = str(k["retained_concepts"]).split("|")
        right_q1 = str(k["q1_key"]).split("|")
        wrong_q1 = [c for c in options_q1 if c not in right_q1] or [f"C{len(options_q1) + 1}"]
        df.at[idx, "answer_q1"] = right_q1[0] if rng.random() < acc else str(rng.choice(wrong_q1))
        df.at[idx, "answer_q2"] = str(k["q2_key"]) if rng.random() < acc else ("no" if k["q2_key"] == "yes" else "yes")
        wrong_q3 = [a for a in Q3_ANSWERS if a != k["q3_key"]]
        df.at[idx, "answer_q3"] = str(k["q3_key"]) if rng.random() < acc else str(rng.choice(wrong_q3))
    return df


# ----------------------------------------------------------------------
# analyze: statistics
# ----------------------------------------------------------------------


def expert_by_format(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Expert by format table of means, restricted to experts with a value for every format."""
    t = df.pivot_table(index="expert_id", columns="format", values=value, aggfunc="mean")
    return t.reindex(columns=FORMAT_ORDER).dropna()


def flatten_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Split the tuple valued confidence interval columns of wilcoxon_holm into low and high columns."""
    df = df.copy()
    for c in [c for c in df.columns if c.endswith("_ci")]:
        df[c.replace("_ci", "_ci_low")] = df[c].map(lambda t: float(t[0]))
        df[c.replace("_ci", "_ci_high")] = df[c].map(lambda t: float(t[1]))
        df = df.drop(columns=c)
    return df


def descriptive_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fmt in FORMAT_ORDER:
        part = df[df["format"] == fmt]
        for r in RATINGS:
            v = part[r].dropna()
            rows.append({"format": fmt, "rating": r, "n": int(v.size), "mean": float(v.mean()) if v.size else np.nan, "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan, "median": float(v.median()) if v.size else np.nan, "n_missing": int(part[r].isna().sum())})
    ratings = pd.DataFrame(rows)
    rows = []
    for fmt in FORMAT_ORDER:
        part = df[df["format"] == fmt]
        for q in QUESTIONS + ["all"]:
            if q == "all":
                v = pd.concat([part[f"correct_{qq}"] for qq in QUESTIONS]).dropna()
                missing = int(sum(part[f"correct_{qq}"].isna().sum() for qq in QUESTIONS))
            else:
                v = part[f"correct_{q}"].dropna()
                missing = int(part[f"correct_{q}"].isna().sum())
            rows.append({"format": fmt, "question": q, "n_answered": int(v.size), "n_correct": int(v.sum()), "accuracy": float(v.mean()) if v.size else np.nan, "n_missing": missing})
    accuracy = pd.DataFrame(rows)
    return ratings, accuracy


def paired_tests(df: pd.DataFrame, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Friedman with Nemenyi and Wilcoxon with Holm on the expert by format means of every outcome."""
    friedman_rows, pair_rows, wilcoxon_frames, notes = [], [], [], []
    for outcome in outcomes:
        table = expert_by_format(df, outcome)
        if table.shape[0] < 2:
            notes.append(f"{outcome}: fewer than two experts with complete data, paired tests skipped")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = S.friedman_nemenyi(table, higher_is_better=True)
            wt = S.wilcoxon_holm(table, reference="fuzzy")
        row = {"outcome": outcome, "n_experts": fr["n_datasets"], "friedman_statistic": fr["friedman_statistic"], "p_value": fr["p_value"], "critical_difference": fr["critical_difference"]}
        for fmt in FORMAT_ORDER:
            row[f"mean_rank_{fmt}"] = float(fr["average_ranks"][fmt])
            row[f"mean_{fmt}"] = float(table[fmt].mean())
        friedman_rows.append(row)
        sig = fr["significant"]
        for a_i, a in enumerate(FORMAT_ORDER):
            for b in FORMAT_ORDER[a_i + 1 :]:
                pair_rows.append({"outcome": outcome, "format_a": a, "format_b": b, "rank_difference": float(abs(fr["average_ranks"][a] - fr["average_ranks"][b])), "critical_difference": fr["critical_difference"], "significant": bool(sig.loc[a, b])})
        wt.insert(0, "outcome", outcome)
        wilcoxon_frames.append(flatten_intervals(wt))
    friedman = pd.DataFrame(friedman_rows)
    pairs = pd.DataFrame(pair_rows)
    wilcoxon = pd.concat(wilcoxon_frames, ignore_index=True) if wilcoxon_frames else pd.DataFrame()
    return friedman, pairs, wilcoxon, notes


def ordinal_tables(df: pd.DataFrame, adjust_expert: bool) -> tuple[pd.DataFrame, str, list[str]]:
    """Proportional odds model of every rating with the format as predictor (reference: fuzzy)."""
    rows, texts, notes = [], [], []
    variants = [("format", ["format"])]
    if adjust_expert:
        variants.append(("format_and_expert", ["format", "expert_id"]))
    for r in RATINGS:
        for variant, predictors in variants:
            d = df[[r] + predictors].dropna().copy()
            if d[r].nunique() < 2 or d.shape[0] < 6:
                notes.append(f"{r} ({variant}): too few observations or a single rating level, ordinal model skipped")
                continue
            d[r] = d[r].astype(int)
            d["format"] = pd.Categorical(d["format"], categories=["fuzzy", "concept", "shap"])
            if "expert_id" in predictors:
                d["expert_id"] = pd.Categorical(d["expert_id"], categories=sorted(d["expert_id"].unique()))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = S.ordinal_model(d, r, predictors)
                ci = res.conf_int()
                for name in res.params.index:
                    rows.append(
                        {
                            "rating": r,
                            "model": variant,
                            "parameter": str(name),
                            "type": "coefficient" if str(name).startswith(("format_", "expert_id_")) else "threshold",
                            "estimate": float(res.params[name]),
                            "std_error": float(res.bse[name]),
                            "z": float(res.tvalues[name]),
                            "p_value": float(res.pvalues[name]),
                            "ci_low": float(ci.loc[name, 0]),
                            "ci_high": float(ci.loc[name, 1]),
                            "odds_ratio": float(np.exp(res.params[name])) if str(name).startswith(("format_", "expert_id_")) else np.nan,
                            "n": int(res.nobs),
                            "converged": bool(getattr(res.mle_retvals, "get", lambda k, d=None: d)("converged", True)),
                            "log_likelihood": float(res.llf),
                        }
                    )
                texts.append(f"Ordinal model of {r} ({variant}); reference format: fuzzy\n" + str(res.summary()) + "\n")
            except Exception as exc:
                notes.append(f"{r} ({variant}): ordinal model failed with {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), "\n".join(texts), notes


def correctness_long(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for q in QUESTIONS:
        part = df[["expert_id", "instance_id", "format", "position", f"correct_{q}"]].rename(columns={f"correct_{q}": "correct"})
        part = part.assign(question=q)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["correct"])
    long["correct"] = long["correct"].astype(float)
    long["fmt_concept"] = (long["format"] == "concept").astype(float)
    long["fmt_shap"] = (long["format"] == "shap").astype(float)
    return long


def mixed_model_tables(long: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Binomial mixed model of correctness, format fixed effect (reference fuzzy), expert random intercept; overall and per question."""
    rows, texts, notes = [], [], []
    subsets = [("all", long)] + [(q, long[long["question"] == q]) for q in QUESTIONS]
    for label, d in subsets:
        d = d.reset_index(drop=True)
        if d.shape[0] < 6 or d["correct"].nunique() < 2 or d["expert_id"].nunique() < 2:
            notes.append(f"correctness ({label}): too few observations, no variation or a single expert, mixed model skipped")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = S.binomial_mixed_model(d, "correct", ["fmt_concept", "fmt_shap"], "expert_id")
            fe_names = list(res.model.exog_names)
            for name, m, s in zip(fe_names, np.asarray(res.fe_mean, dtype=float), np.asarray(res.fe_sd, dtype=float)):
                rows.append({"subset": label, "parameter": str(name), "type": "fixed", "posterior_mean": float(m), "posterior_sd": float(s), "z": float(m / s) if s > 0 else np.nan, "odds_ratio": float(np.exp(m)), "n": int(d.shape[0]), "n_experts": int(d["expert_id"].nunique())})
            for name, m, s in zip(list(res.model.vcp_names), np.asarray(res.vcp_mean, dtype=float), np.asarray(res.vcp_sd, dtype=float)):
                rows.append({"subset": label, "parameter": f"log_sd_random_intercept_{name}", "type": "variance", "posterior_mean": float(m), "posterior_sd": float(s), "z": np.nan, "odds_ratio": np.nan, "n": int(d.shape[0]), "n_experts": int(d["expert_id"].nunique())})
            texts.append(f"Binomial mixed model of correctness ({label}); fixed effect format with reference fuzzy, random intercept per expert\n" + str(res.summary()) + "\n")
        except Exception as exc:
            notes.append(f"correctness ({label}): mixed model failed with {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), "\n".join(texts), notes


def plot_ratings(df: pd.DataFrame, path: Path) -> None:
    """Expert means per format for every rating, with the overall mean as a horizontal mark."""
    plt = plotting._plt()
    fig, axes = plt.subplots(1, len(RATINGS), figsize=(3.0 * len(RATINGS), 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, r in zip(np.atleast_1d(axes), RATINGS):
        t = expert_by_format(df, r)
        for j, fmt in enumerate(FORMAT_ORDER):
            x = j + rng.uniform(-0.12, 0.12, size=t.shape[0])
            ax.scatter(x, t[fmt], s=16, color=PALETTE[j], alpha=0.8, linewidth=0)
            ax.hlines(float(df.loc[df["format"] == fmt, r].mean()), j - 0.3, j + 0.3, color=TEXT, linewidth=1.5)
        ax.set_xticks(range(len(FORMAT_ORDER)))
        ax.set_xticklabels(FORMAT_ORDER)
        ax.set_title(r.replace("rating_", "").replace("_", " "), loc="left", fontsize=9)
        ax.set_ylim(0.5, 7.5)
        style_axes(ax)
    np.atleast_1d(axes)[0].set_ylabel("expert mean rating")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, simulated: bool, df: pd.DataFrame, ratings: pd.DataFrame, accuracy: pd.DataFrame, friedman: pd.DataFrame, wilcoxon: pd.DataFrame, ordinal: pd.DataFrame, mixed: pd.DataFrame, notes: list[str]) -> None:
    lines = ["Phase D: expert judgment study, analysis summary", ""]
    if simulated:
        lines += ["SIMULATED RESPONSES. The response sheet was filled with random values by analyze --simulate to test the analysis path.", "The numbers below carry no evidence about the formats.", ""]
    lines.append(f"Experts: {df['expert_id'].nunique()}, predictions: {df['instance_id'].nunique()}, items: {len(df)}, formats: {', '.join(FORMAT_ORDER)} (reference: fuzzy)")
    lines.append("")
    lines.append("Ratings per format (mean, sd, median, n)")
    for r in RATINGS:
        part = ratings[ratings["rating"] == r]
        lines.append("  " + r + ": " + "; ".join(f"{row['format']} {row['mean']:.2f} ({row['sd']:.2f}, {row['median']:.1f}, n={row['n']})" for _, row in part.iterrows()))
    lines.append("")
    lines.append("Accuracy per format (proportion correct, n answered)")
    for q in QUESTIONS + ["all"]:
        part = accuracy[accuracy["question"] == q]
        lines.append(f"  {q}: " + "; ".join(f"{row['format']} {row['accuracy']:.2f} (n={row['n_answered']})" for _, row in part.iterrows()))
    lines.append("")
    if len(friedman):
        lines.append("Friedman test across the three formats on expert means (mean ranks, higher rating is better)")
        for _, row in friedman.iterrows():
            ranks = ", ".join(f"{fmt} {row[f'mean_rank_{fmt}']:.2f}" for fmt in FORMAT_ORDER)
            lines.append(f"  {row['outcome']}: chi2 = {row['friedman_statistic']:.3f}, p = {row['p_value']:.4f}, Nemenyi critical difference {row['critical_difference']:.2f}, ranks {ranks}")
        lines.append("")
    if len(wilcoxon):
        lines.append("Wilcoxon signed rank tests against the fuzzy format (Holm adjusted p, Cliff's delta of format minus fuzzy)")
        for _, row in wilcoxon.iterrows():
            lines.append(f"  {row['outcome']}: {row['method']} vs fuzzy, p = {row['p_value']:.4f}, p Holm = {row['p_holm']:.4f}, mean difference {row['mean_difference']:+.2f}, Cliff's delta {row['cliff_delta']:+.2f}")
        lines.append("")
    if len(ordinal):
        lines.append("Ordinal (proportional odds) models per rating, format only, coefficients relative to fuzzy (log odds of a higher rating)")
        part = ordinal[(ordinal["model"] == "format") & (ordinal["type"] == "coefficient")]
        for _, row in part.iterrows():
            lines.append(f"  {row['rating']}: {row['parameter']} = {row['estimate']:+.3f} (se {row['std_error']:.3f}, p = {row['p_value']:.4f}, odds ratio {row['odds_ratio']:.2f})")
        lines.append("")
    if len(mixed):
        lines.append("Binomial mixed model of correctness (posterior means, fixed effects relative to fuzzy, random intercept per expert)")
        for _, row in mixed[mixed["type"] == "fixed"].iterrows():
            lines.append(f"  {row['subset']}: {row['parameter']} = {row['posterior_mean']:+.3f} (sd {row['posterior_sd']:.3f}, odds ratio {row['odds_ratio']:.2f})")
        lines.append("")
    if notes:
        lines.append("Notes")
        lines += ["  " + n for n in notes]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run_analyze(args) -> Path:
    run_dir = Path(args.results) / args.name
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"No study materials found in {run_dir}; run the generate subcommand first.")
    cfg = load_config(cfg_path)
    priv_dir = run_dir / "private"
    mapping = pd.read_csv(priv_dir / "format_mapping.csv", dtype=str)
    key = pd.read_csv(priv_dir / "answer_key.csv")
    likert = (int(cfg.get("likert_min", 1)), int(cfg.get("likert_max", 7)))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    if args.simulate:
        sheet = read_responses([run_dir / "response_sheet.csv"])
        rng = np.random.default_rng(seed + 200)
        responses = simulate_responses(sheet, mapping, key, cfg.get("simulation", {}), likert, rng)
        resp_path = run_dir / "responses_simulated.csv"
        responses.to_csv(resp_path, index=False)
        print(f"[phase D] analyze: simulated {len(responses)} responses with seed {seed + 200} into {resp_path}")
        paths = [resp_path]
    else:
        paths = [Path(p) for p in (args.responses or [run_dir / "responses.csv"])]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Response files not found: {missing}. Pass the filled sheets with --responses, or use --simulate to test the analysis path.")
    out_dir = run_dir / (args.out or ("analysis_simulated" if args.simulate else "analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = score_responses(read_responses(paths), mapping, key, likert)
    print(f"[phase D] analyze: {df['expert_id'].nunique()} experts, {df['instance_id'].nunique()} predictions, {len(df)} items, outputs in {out_dir}")
    df.to_csv(out_dir / "scored_responses.csv", index=False)
    notes: list[str] = []
    complete = df.groupby("expert_id")["format"].nunique()
    incomplete = complete[complete < len(FORMAT_ORDER)].index.tolist()
    if incomplete:
        notes.append(f"experts without all three formats (excluded from the paired tests): {incomplete}")

    ratings, accuracy = descriptive_tables(df)
    ratings.to_csv(out_dir / "descriptives_ratings.csv", index=False)
    accuracy.to_csv(out_dir / "descriptives_accuracy.csv", index=False)
    friedman, pairs, wilcoxon, n1 = paired_tests(df, RATINGS + ["accuracy"])
    notes += n1
    friedman.to_csv(out_dir / "friedman.csv", index=False)
    pairs.to_csv(out_dir / "nemenyi_pairs.csv", index=False)
    wilcoxon.to_csv(out_dir / "wilcoxon_vs_fuzzy.csv", index=False)
    ordinal, ordinal_text, n2 = ordinal_tables(df, bool(cfg.get("ordinal_adjust_expert", True)))
    notes += n2
    ordinal.to_csv(out_dir / "ordinal_models.csv", index=False)
    (out_dir / "ordinal_models.txt").write_text(ordinal_text, encoding="utf8")
    long = correctness_long(df)
    long.to_csv(out_dir / "correctness_long.csv", index=False)
    mixed, mixed_text, n3 = mixed_model_tables(long)
    notes += n3
    mixed.to_csv(out_dir / "accuracy_mixed_model.csv", index=False)
    (out_dir / "accuracy_mixed_model.txt").write_text(mixed_text, encoding="utf8")
    try:
        plot_ratings(df, out_dir / "ratings_by_format.png")
    except Exception as exc:
        notes.append(f"ratings figure failed with {type(exc).__name__}: {exc}")
    write_summary(out_dir / "summary.txt", bool(args.simulate), df, ratings, accuracy, friedman, wilcoxon, ordinal, mixed, notes)
    for n in notes:
        print("[phase D] note: " + n)
    print((out_dir / "summary.txt").read_text(encoding="utf8"))
    print("[phase D] analyze finished")
    return out_dir


# ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="fit PFCA, render the three formats and write the response sheets and the answer key")
    g.add_argument("--config", default="experiments/configs/phase_d.yaml")
    g.add_argument("--results", default="results")
    g.add_argument("--name", default="phase_d")
    g.add_argument("--synthetic", action="store_true", help="use the synthetic problem of the configuration instead of the questionnaire")
    g.add_argument("--csv", default=None, help="override csv_path of the configuration (questionnaire)")
    g.add_argument("--spec", default=None, help="override spec_path of the configuration (questionnaire)")
    g.add_argument("--n-instances", type=int, default=None, help="predictions shown to every expert")
    g.add_argument("--n-experts", type=int, default=None)
    g.add_argument("--n-candidates", type=int, default=None, help="instances explained before the presented ones are chosen")
    g.add_argument("--n-resamples", type=int, default=None)
    g.add_argument("--n-jobs", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    a = sub.add_parser("analyze", help="score the filled response sheets and run the statistical analysis")
    a.add_argument("--results", default="results")
    a.add_argument("--name", default="phase_d")
    a.add_argument("--responses", nargs="+", default=None, help="filled response sheet CSV files (default results/<name>/responses.csv); concatenated files are accepted")
    a.add_argument("--simulate", action="store_true", help="fill the blank response sheet of the run with random plausible responses and analyze those")
    a.add_argument("--seed", type=int, default=None, help="seed of the simulation (default: the seed of the run)")
    a.add_argument("--out", default=None, help="name of the output folder inside the run directory")
    args = ap.parse_args(argv)
    if args.command == "generate":
        run_generate(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
