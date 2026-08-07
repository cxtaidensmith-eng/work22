"""Shared, leakage-conscious utilities for TPCF v1."""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ("AD", "CN", "SMCI")
CLASS_IDS = np.arange(3, dtype=np.int64)
PROBABILITY_COLUMNS = tuple(f"probability_{name}" for name in CLASS_ORDER)
ORIGINAL_GIT_SPEC = (
    "refs/remotes/origin/experiment/um-ler-frozen-v2:"
    "results/um_ler_frozen_v2/original_cache/original_oof_predictions.csv"
)
DEFAULT_TABPFN_OOF = (
    ROOT
    / "experiments/tabpfn_bounded_tuning_v1/e4/formal/oof_predictions.csv"
)
EXPECTED_ORIGINAL = {
    "correct": 556,
    "acc": 0.9297658862876255,
    "macro_f1": 0.9140778251271361,
    "bacc": 0.9140778251271361,
    "macro_auc": 0.9560490697782983,
    "weighted_f1": 0.9297658862876255,
    "confusion_matrix": [[62, 0, 10], [0, 198, 11], [10, 11, 296]],
}
EXPECTED_TABPFN = {
    "correct": 548,
    "acc": 0.9163879598662207,
    "macro_f1": 0.9058959712564674,
    "bacc": 0.9013186544732287,
    "macro_auc": 0.9764191581686475,
    "weighted_f1": 0.9163072846281483,
    "confusion_matrix": [[62, 0, 10], [0, 190, 19], [8, 13, 296]],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def unique_directory(preferred: Path) -> Path:
    if not preferred.exists():
        return preferred
    suffix = 2
    while True:
        candidate = preferred.with_name(f"{preferred.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def probability_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    return -(probability * np.log(probability)).sum(axis=1)


def probability_margin(probability: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probability, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(left, dtype=np.float64), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=np.float64), 1e-12, 1.0)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (np.log(left) - np.log(middle))).sum(axis=1)
        + (right * (np.log(right) - np.log(middle))).sum(axis=1)
    )


def gate_features(original: np.ndarray, tabpfn: np.ndarray) -> np.ndarray:
    original_prediction = np.argmax(original, axis=1)
    tabpfn_prediction = np.argmax(tabpfn, axis=1)
    return np.column_stack(
        [
            probability_entropy(original),
            probability_entropy(tabpfn),
            probability_margin(original),
            probability_margin(tabpfn),
            js_divergence(original, tabpfn),
            (original_prediction != tabpfn_prediction).astype(np.float64),
        ]
    )


def validate_probability(probability: np.ndarray, name: str, rows: int = 598) -> None:
    probability = np.asarray(probability, dtype=np.float64)
    require(probability.shape == (rows, 3), f"{name}: probability shape {probability.shape}")
    require(np.isfinite(probability).all(), f"{name}: probability contains NaN/Inf")
    require(np.all(probability >= 0.0), f"{name}: probability contains negatives")
    require(
        np.allclose(probability.sum(axis=1), 1.0, atol=2e-7, rtol=0.0),
        f"{name}: probability rows do not sum to one",
    )


def metric_bundle(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    validate_probability(probability, "metric input", rows=truth.size)
    prediction = np.argmax(probability, axis=1).astype(np.int64)
    return {
        "correct": int((prediction == truth).sum()),
        "acc": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_IDS, average="macro", zero_division=0)),
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "macro_auc": float(roc_auc_score(truth, probability, labels=CLASS_IDS, multi_class="ovr", average="macro")),
        "weighted_f1": float(f1_score(truth, prediction, labels=CLASS_IDS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=CLASS_IDS).astype(int).tolist(),
    }


def assert_metrics(actual: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    require(actual["correct"] == expected["correct"], f"{name}: correct baseline mismatch")
    for key in ("acc", "macro_f1", "bacc", "macro_auc", "weighted_f1"):
        require(
            math.isclose(float(actual[key]), float(expected[key]), rel_tol=0.0, abs_tol=2e-10),
            f"{name}: {key} baseline mismatch: {actual[key]} vs {expected[key]}",
        )
    require(actual["confusion_matrix"] == expected["confusion_matrix"], f"{name}: confusion mismatch")


def _read_original(explicit_path: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        require(path.is_file(), f"Original OOF missing: {path}")
        return pd.read_csv(path, float_precision="round_trip"), {
            "kind": "file",
            "path": str(path),
            "sha256": file_sha256(path),
        }
    raw = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-C",
            str(ROOT),
            "show",
            ORIGINAL_GIT_SPEC,
        ],
        stderr=subprocess.STDOUT,
    )
    return pd.read_csv(io.BytesIO(raw), float_precision="round_trip"), {
        "kind": "git_object",
        "spec": ORIGINAL_GIT_SPEC,
        "sha256": bytes_sha256(raw),
    }


def load_aligned_outer_oof(
    original_path: str | None = None,
    tabpfn_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original, original_source = _read_original(original_path)
    tab_path = Path(tabpfn_path).expanduser().resolve() if tabpfn_path else DEFAULT_TABPFN_OOF
    require(tab_path.is_file(), f"TabPFN E4 OOF missing: {tab_path}")
    tabpfn = pd.read_csv(tab_path, float_precision="round_trip")
    required = {"subject_index", "fold", "truth", "prediction", *PROBABILITY_COLUMNS}
    for name, frame in (("Original", original), ("TabPFN E4", tabpfn)):
        require(required.issubset(frame.columns), f"{name}: missing OOF columns")
        require(len(frame) == 598, f"{name}: expected 598 rows")
        require(frame["subject_index"].nunique() == 598, f"{name}: subjects not unique")
        require(set(frame["fold"].astype(int)) == set(range(10)), f"{name}: fold coverage mismatch")
    original = original.sort_values("subject_index").reset_index(drop=True)
    tabpfn = tabpfn.sort_values("subject_index").reset_index(drop=True)
    for column in ("subject_index", "fold", "truth"):
        require(
            np.array_equal(original[column].to_numpy(), tabpfn[column].to_numpy()),
            f"Outer OOF alignment failed: {column}",
        )
    require(
        np.array_equal(original["subject_index"].to_numpy(dtype=np.int64), np.arange(598)),
        "Expected subject_index 0..597 after alignment",
    )
    original_probability = original[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    tabpfn_probability = tabpfn[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    validate_probability(original_probability, "Original")
    validate_probability(tabpfn_probability, "TabPFN E4")
    require(
        np.array_equal(np.argmax(original_probability, axis=1), original["prediction"].to_numpy(dtype=np.int64)),
        "Original prediction is not softmax argmax",
    )
    require(
        np.array_equal(np.argmax(tabpfn_probability, axis=1), tabpfn["prediction"].to_numpy(dtype=np.int64)),
        "TabPFN prediction is not probability argmax",
    )
    truth = original["truth"].to_numpy(dtype=np.int64)
    original_metrics = metric_bundle(truth, original_probability)
    tabpfn_metrics = metric_bundle(truth, tabpfn_probability)
    assert_metrics(original_metrics, EXPECTED_ORIGINAL, "Original")
    assert_metrics(tabpfn_metrics, EXPECTED_TABPFN, "TabPFN E4")
    aligned = pd.DataFrame(
        {
            "subject_index": original["subject_index"].to_numpy(dtype=np.int64),
            "fold": original["fold"].to_numpy(dtype=np.int64),
            "truth": truth,
            "original_prediction": np.argmax(original_probability, axis=1),
            "tabpfn_prediction": np.argmax(tabpfn_probability, axis=1),
        }
    )
    for index, class_name in enumerate(CLASS_ORDER):
        aligned[f"original_probability_{class_name}"] = original_probability[:, index]
        aligned[f"tabpfn_probability_{class_name}"] = tabpfn_probability[:, index]
    provenance = {
        "class_order": list(CLASS_ORDER),
        "alignment_key": "subject_index",
        "rows": 598,
        "unique_subjects": 598,
        "fold_truth_aligned": True,
        "original_source": original_source,
        "tabpfn_source": {
            "kind": "file",
            "path": str(tab_path),
            "sha256": file_sha256(tab_path),
        },
        "original_metrics": original_metrics,
        "tabpfn_metrics": tabpfn_metrics,
    }
    return aligned, provenance


def aligned_probabilities(aligned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = aligned["truth"].to_numpy(dtype=np.int64)
    original = aligned[[f"original_probability_{name}" for name in CLASS_ORDER]].to_numpy(dtype=np.float64)
    tabpfn = aligned[[f"tabpfn_probability_{name}" for name in CLASS_ORDER]].to_numpy(dtype=np.float64)
    return truth, original, tabpfn


def comparison_with_original(
    truth: np.ndarray,
    original_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, Any]:
    original_prediction = np.argmax(original_probability, axis=1)
    candidate_prediction = np.argmax(candidate_probability, axis=1)
    original_correct = original_prediction == truth
    candidate_correct = candidate_prediction == truth
    changed = candidate_prediction != original_prediction
    repair = (~original_correct) & candidate_correct
    damage = original_correct & (~candidate_correct)
    by_class = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {
            "repairs": int((repair & mask).sum()),
            "damages": int((damage & mask).sum()),
        }
    return {
        "changed_predictions": int(changed.sum()),
        "repairs": int(repair.sum()),
        "damages": int(damage.sum()),
        "original_only_correct": int((original_correct & ~candidate_correct).sum()),
        "candidate_only_correct": int((~original_correct & candidate_correct).sum()),
        "by_true_class": by_class,
    }


def complementarity(
    truth: np.ndarray,
    original_probability: np.ndarray,
    tabpfn_probability: np.ndarray,
) -> dict[str, Any]:
    original_prediction = np.argmax(original_probability, axis=1)
    tabpfn_prediction = np.argmax(tabpfn_probability, axis=1)
    original_correct = original_prediction == truth
    tabpfn_correct = tabpfn_prediction == truth
    both = original_correct & tabpfn_correct
    original_only = original_correct & ~tabpfn_correct
    tabpfn_only = ~original_correct & tabpfn_correct
    neither = ~original_correct & ~tabpfn_correct
    by_class = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        mask = truth == class_id
        by_class[class_name] = {
            "support": int(mask.sum()),
            "both_correct": int((both & mask).sum()),
            "original_only_correct": int((original_only & mask).sum()),
            "tabpfn_only_correct": int((tabpfn_only & mask).sum()),
            "both_wrong": int((neither & mask).sum()),
            "repairs_available": int((tabpfn_only & mask).sum()),
            "damages_at_risk": int((original_only & mask).sum()),
        }
    return {
        "both_correct": int(both.sum()),
        "original_only_correct": int(original_only.sum()),
        "tabpfn_only_correct": int(tabpfn_only.sum()),
        "both_wrong": int(neither.sum()),
        "oracle_union_correct": int((original_correct | tabpfn_correct).sum()),
        "by_true_class": by_class,
    }


def exact_mcnemar(original_only: int, candidate_only: int) -> float:
    discordant = int(original_only + candidate_only)
    if discordant == 0:
        return 1.0
    return float(binomtest(min(original_only, candidate_only), discordant, 0.5).pvalue)
