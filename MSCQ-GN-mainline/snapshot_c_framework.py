"""Archive/verify existing C evidence. Never trains or changes model code.

The archive includes logs, predictions, protocols and source snapshots. Large
datasets/checkpoints remain local and are recorded by SHA-256, not copied.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "experiment_snapshots/c_framework_20260929"
RESULT = ROOT / "Result"

EXPERIMENTS = [
    "relation_mechanism_nodecay", "c_result_reaudit_20260924T015759Z",
    "c_mechanism_controls_seed0_20260924", "c_d3_fuse_first_seed0_20260925",
    "c_cross_relation_null_seed0_20260928", "branch_fusion_seed0_20260924",
    "relation_aggregation_seed0_20260924", "alternating_cooperative_seed0_20260923",
    "alternating_a1_seeds12_20260924", "grouped_content_seed0_20260929",
    "c_readout_diagnosis_20260922", "c_current_readout_20260928",
    "c_tadpole_binary_seed0_20260924", "tad_binary_fine_tune_20260928",
    "tad_binary_joint_tune_20260928", "tad_binary_loss_tune_20260928",
    "abide_fine_tune_20260927", "abide_noise_adaptation_20260927",
    "abide_n1_followup_20260927", "c_abide_repair_v2_20260927", "c_abide5_seed0_20260924",
    "abide5_fine_tune_20260928", "abide5_tie_completion_20260928",
]


def groups():
    rows = []

    def add(task, variant, seed, base):
        rows.append(dict(task=task, variant=variant, seed=seed, source=base))

    for seed in range(3):
        for variant, mode in [("C", "content"), ("static_relation_content", "static")]:
            add("TAD3", variant, seed, f"relation_mechanism_nodecay/seed_{seed}/{mode}")
    for experiment, variants in [
        ("c_mechanism_controls_seed0_20260924", ["D1", "D2"]),
        ("c_d3_fuse_first_seed0_20260925", ["D3"]),
        ("c_cross_relation_null_seed0_20260928", ["XREL_NULL"]),
        ("branch_fusion_seed0_20260924", ["concat", "wide"]),
        ("relation_aggregation_seed0_20260924", ["R1", "R2"]),
        ("grouped_content_seed0_20260929", ["G2", "W38"]),
    ]:
        for variant in variants:
            add("TAD3", variant, 0, f"{experiment}/seed_0/{variant}")
    add("TAD3", "A1", 0, "alternating_cooperative_seed0_20260923/seed_0/A1")
    for seed in (1, 2):
        add("TAD3", "A1", seed, f"alternating_a1_seeds12_20260924/seed_{seed}/A1")
    add("TAD2", "C_J_DN", 0, "tad_binary_joint_tune_20260928/runs/J_DN")
    add("ABIDE", "C_R_L115", 0, "abide_fine_tune_20260927/runs/R_L115")
    add("ABIDE5", "C_BASE", 0, "c_abide5_seed0_20260924/content")
    return rows


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def project_files():
    files = {p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".py", ".ps1", ".md", ".txt"}}
    files.add(ROOT / ".gitignore")
    files.add(ROOT / ".gitattributes")
    for directory in ("Model", "Utils", "Loss", "tests", "Config"):
        files.update(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".ini"} and "__pycache__" not in p.parts)
    files.update(p for p in (ROOT / "reference_fixed10").glob("*") if p.is_file())
    files.update(p for p in (ROOT / "reference_fixed10/provenance").glob("*.json"))
    for relative in [
        "DATASET/TADPOLE/AD_CN_SMCI_ADNI_modal_feat_dict.npy",
        "DATASET/TADPOLE/SMCI_PMCI_ADNI_modal_feat_dict.npy",
        "DATASET/ABIDE/ABIDE_modal_feat_dict.npy",
        "DATASET/ABIDE-5/ABIDE_modal_feat_dict.npy",
    ]:
        files.add(ROOT / relative)
    return sorted(files)


def archive_evidence():
    copied = []

    def copy(path):
        relative = path.relative_to(RESULT)
        destination = ARCHIVE / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha(destination) != sha(path):
                raise RuntimeError(f"Refusing to overwrite changed archive file: {destination}")
        else:
            shutil.copy2(path, destination)
        copied.append(dict(source=path.relative_to(ROOT).as_posix(),
                           archive=destination.relative_to(ARCHIVE).as_posix(), sha256=sha(path)))

    selected_folders = {
        (RESULT / g["source"] / f"fold_{fold:02d}").resolve()
        for g in groups() for fold in range(1, 11)
    }
    for name in EXPERIMENTS:
        base = RESULT / name
        if not base.is_dir():
            raise FileNotFoundError(base)
        for p in sorted(base.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.suffix not in {".json", ".md", ".csv", ".ini", ".npz", ".py", ".ps1", ".txt"}:
                continue
            fold_ancestor = next((q for q in p.parents if re.fullmatch(r"fold_\d+", q.name)), None)
            if fold_ancestor is not None:
                if fold_ancestor.resolve() not in selected_folders:
                    continue
                if p.name not in {"complete.json", "best.json", "last.json", "epochs.csv",
                                   "predictions.npz", "best_predictions.npz", "last_predictions.npz",
                                   "effective_settings.json", "effective_loss.json", "optimizer_groups.json", "fine_audit.json",
                                   "diagnostics_summary.json"}:
                    continue
            copy(p)
    for p in sorted(RESULT.iterdir()):
        if p.is_file() and p.suffix in {".json", ".md"} and any(
            key in p.name for key in ("TAD二分类", "ABIDE-5", "分组内容传播")
        ):
            copy(p)
    return copied


def score_npz(path, reference_rows=None):
    with np.load(path, allow_pickle=False) as z:
        y, logits = z["labels"], z["logits"]
        rows = z["row_index"] if "row_index" in z else reference_rows
        if rows is None:
            raise ValueError(f"Missing row mapping: {path}")
        assert np.isfinite(logits).all() and len(rows) == len(y) == len(logits)
        assert len(np.unique(rows)) == len(rows)
        pred = logits.argmax(1)
        labels = np.arange(logits.shape[1])
        recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        return dict(acc=float(np.mean(pred == y)),
                    macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
                    balanced_acc=float(np.mean(recalls)), per_class_recall=recalls.tolist(),
                    confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist()), rows.copy(), y.copy(), pred.copy()


def recompute():
    summaries, fold_rows, predictions, paired = [], [], {}, []
    for g in groups():
        entries, all_rows, all_y, all_pred = [], [], [], []
        for fold in range(1, 11):
            folder = ARCHIVE / "evidence" / g["source"] / f"fold_{fold:02d}"
            record = load(folder / "complete.json")
            assert record["fold"] == fold, folder
            if "seed" in record:
                assert record["seed"] == g["seed"], folder
            assert record["last"]["epoch"] == 1000, folder
            best_path = folder / "predictions.npz"
            if not best_path.exists():
                best_path = folder / "best_predictions.npz"
            best, rows, y, pred = score_npz(best_path)
            last, last_rows, last_y, _ = score_npz(folder / "last_predictions.npz", rows)
            assert np.array_equal(rows, last_rows) and np.array_equal(y, last_y)
            with (folder / "epochs.csv").open(encoding="utf-8-sig", newline="") as stream:
                logs = list(csv.DictReader(stream))
            assert [int(row["epoch"]) for row in logs] == list(range(1, 1001)), folder
            acc = np.array([float(row["test_acc"]) for row in logs])
            chosen_epoch = int(acc.argmax()) + 1
            assert chosen_epoch == record["best"]["epoch"], (folder, chosen_epoch, record["best"])
            assert abs(best["acc"] - acc.max()) < 1e-7, folder
            assert abs(last["acc"] - acc[-1]) < 1e-7, folder
            for checkpoint, measured in [("best", best), ("last", last)]:
                logged = record[checkpoint]
                for field in ("acc", "macro_f1"):
                    original = logged.get(field, logged.get("test_acc") if field == "acc" else None)
                    if original is not None:
                        assert abs(measured[field] - original) < 1e-7, (folder, checkpoint, field)
            entry = dict(task=g["task"], variant=g["variant"], seed=g["seed"], fold=fold,
                         parameters=record["parameters"], samples=len(y), best_epoch=chosen_epoch,
                         best_acc=best["acc"], best_macro_f1=best["macro_f1"],
                         best_balanced_acc=best["balanced_acc"], last_acc=last["acc"],
                         last100_acc=float(acc[-100:].mean()),
                         best_confusion_matrix=best["confusion_matrix"],
                         source=folder.relative_to(ARCHIVE).as_posix())
            entries.append(entry)
            fold_rows.append(entry)
            all_rows.extend(rows.tolist())
            all_y.extend(y.tolist())
            all_pred.extend(pred.tolist())
        expected_n = {"TAD3": 598, "TAD2": 535, "ABIDE": 871, "ABIDE5": 864}[g["task"]]
        assert len(all_rows) == len(set(all_rows)) == expected_n
        assert set(all_rows) == set(range(expected_n)), g
        assert len({e["parameters"] for e in entries}) == 1, g
        summary = {key: g[key] for key in ("task", "variant", "seed", "source")}
        summary.update(samples=expected_n, folds=10, epochs_per_fold=1000,
                       parameters=entries[0]["parameters"],
                       best_epoch_by_fold=[e["best_epoch"] for e in entries])
        for field in ("best_acc", "best_macro_f1", "best_balanced_acc", "last_acc", "last100_acc"):
            summary[field] = float(np.mean([e[field] for e in entries]))
        summary["pooled_confusion_matrix"] = np.sum([e["best_confusion_matrix"] for e in entries], axis=0).tolist()
        summaries.append(summary)
        predictions[(g["task"], g["variant"], g["seed"])] = (np.array(all_rows), np.array(all_y), np.array(all_pred))
    for g in summaries:
        if g["task"] != "TAD3" or g["variant"] == "C":
            continue
        rows, y, pred = predictions[(g["task"], g["variant"], g["seed"])]
        cr, cy, cp = predictions[("TAD3", "C", g["seed"])]
        assert np.array_equal(rows, cr) and np.array_equal(y, cy)
        ccorrect, vcorrect = cp == y, pred == y
        baseline = next(s for s in summaries if s["task"] == "TAD3" and s["variant"] == "C" and s["seed"] == g["seed"])
        paired.append(dict(variant=g["variant"], seed=g["seed"],
                           delta_best_acc_pp=100 * (g["best_acc"] - baseline["best_acc"]),
                           corrected=int((~ccorrect & vcorrect).sum()), harmed=int((ccorrect & ~vcorrect).sum())))
    return dict(metric_unit="fraction, except delta_best_acc_pp; equal weight across folds",
                selection="first maximum TEST ACC among 1000 epochs, original exploratory protocol",
                summaries=summaries, folds=fold_rows, paired_vs_c=paired)


def external_assets():
    paths = {p for p in (ROOT / "DATASET").rglob("*") if p.is_file() and p not in set(project_files())}
    for g in groups():
        for fold in range(1, 11):
            folder = RESULT / g["source"] / f"fold_{fold:02d}"
            paths.update(folder.glob("*.pt"))
    assets = []
    for p in sorted(paths):
        assets.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                           sha256=sha(p), category="checkpoint" if p.suffix == ".pt" else "dataset"))
    return dict(note="These files are NOT copied into Git. Local-only paths/hashes cannot recover deleted assets. "
                     "Original TAD3 content/static runs have best.pt only, not last.pt.", files=assets)


def write_manifest():
    paths = project_files() + [p for p in ARCHIVE.rglob("*") if p.is_file() and p.name not in {"manifest.json", "verification.json"}]
    records = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(paths))]
    dump(ARCHIVE / "manifest.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
                                          project_root=str(ROOT), files=records,
                                          total_bytes=sum(r["bytes"] for r in records)))


def verify(check_originals=False, check_external=False):
    manifest = load(ARCHIVE / "manifest.json")
    for record in manifest["files"]:
        p = ROOT / record["path"]
        assert p.is_file() and p.stat().st_size == record["bytes"] and sha(p) == record["sha256"], p
    metrics = recompute()
    assert metrics == load(ARCHIVE / "metrics.json"), "Archived metrics differ from recomputed predictions/logs"
    if check_originals:
        for record in load(ARCHIVE / "provenance.json")["files"]:
            assert sha(ROOT / record["source"]) == record["sha256"], record["source"]
            assert sha(ARCHIVE / record["archive"]) == record["sha256"], record["archive"]
    if check_external:
        for record in load(ARCHIVE / "external_assets.json")["files"]:
            assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
    report = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
                  hash_verified_files=len(manifest["files"]),
                  recomputed_folds=len(metrics["folds"]), prediction_files=2*len(metrics["folds"]),
                  epoch_rows=1000*len(metrics["folds"]),
                  original_evidence_checked=check_originals, external_assets_checked=check_external,
                  checks="SHA-256; contiguous 1000 epochs; first maximum TEST ACC; best/last predictions; "
                         "stored metrics; 10-fold OOF coverage; paired row/label alignment",
                  scope="Archive and metrics verification only. No new training or checkpoint inference performed.")
    dump(ARCHIVE / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build():
    if (ARCHIVE / "manifest.json").exists():
        raise FileExistsError("Snapshot already finalized. Use --verify; do not silently overwrite it.")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not (ARCHIVE / "README.md").exists():
        raise FileNotFoundError("Write the snapshot README before building the archive")
    copied = archive_evidence()
    dump(ARCHIVE / "provenance.json", dict(note="Byte-for-byte copies of pre-existing experimental evidence", files=copied))
    metrics = recompute()
    dump(ARCHIVE / "metrics.json", metrics)
    with (ARCHIVE / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [k for k in metrics["summaries"][0] if k not in {"pooled_confusion_matrix", "best_epoch_by_fold"}]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics["summaries"])
    with (ARCHIVE / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [k for k in metrics["folds"][0] if k != "best_confusion_matrix"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics["folds"])
    dump(ARCHIVE / "external_assets.json", external_assets())
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "tqdm", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dump(ARCHIVE / "environment.json", dict(python=sys.version, executable=sys.executable,
                                           platform=platform.platform(), packages=packages,
                                           note="Environment at snapshot time; original per-experiment metadata retained in evidence."))
    (ARCHIVE / "requirements-snapshot.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items() if v) + "\n", encoding="utf-8")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    dump(ARCHIVE / "git_context_before_save.json", dict(parent_head=git("rev-parse", "HEAD"),
                                                       parent_branch=git("branch", "--show-current"),
                                                       scope="Only MSCQ-GN-mainline curated files; existing parent deletions/edits excluded.",
                                                       status=git("status", "--porcelain", "--untracked-files=no").splitlines()))
    write_manifest()
    verify(check_originals=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--refresh-manifest", action="store_true", help="After documentation-only edits before the first Git commit")
    parser.add_argument("--check-originals", action="store_true")
    parser.add_argument("--check-external", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    elif args.refresh_manifest:
        write_manifest()
        verify(check_originals=args.check_originals, check_external=args.check_external)
    else:
        verify(check_originals=args.check_originals, check_external=args.check_external)
