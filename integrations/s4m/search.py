from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


UPSTREAM_COMMIT = "a718823addd3606e763dfc261174e0135b2535f4"


def build_s4m_subprocess_environment(
    s4m_root: Path,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    model_path = str((s4m_root.resolve() / "s4m" / "model"))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        model_path + os.pathsep + existing if existing else model_path
    )
    return environment


def _number_slug(value: float) -> str:
    return format(value, "g").replace(".", "p")


@dataclass(frozen=True)
class SearchCandidate:
    e_layers: int
    learning_rate: float
    dropout: float
    profile: str

    @property
    def run_id(self) -> str:
        return self.run_id_for("missing")

    def run_id_for(self, variant: str) -> str:
        return (
            f"S4M-ETTh1-{variant}-el{self.e_layers}"
            f"-lr{_number_slug(self.learning_rate)}"
            f"-do{_number_slug(self.dropout)}-adam"
        )


def official_candidates() -> list[SearchCandidate]:
    return [
        SearchCandidate(
            e_layers=e_layers,
            learning_rate=0.005,
            dropout=0.1,
            profile="official",
        )
        for e_layers in (2, 4, 8)
    ]


def tuned_stage1_candidates() -> list[SearchCandidate]:
    return [
        SearchCandidate(
            e_layers=e_layers,
            learning_rate=learning_rate,
            dropout=0.1,
            profile="tuned-stage1",
        )
        for e_layers, learning_rate in itertools.product(
            (2, 4, 8), (0.001, 0.003, 0.005)
        )
    ]


def dropout_refinement_candidates(
    candidates: Sequence[SearchCandidate],
    validation_mse: dict[str, float],
) -> list[SearchCandidate]:
    missing_scores = [
        candidate.run_id
        for candidate in candidates
        if candidate.run_id not in validation_mse
    ]
    if missing_scores:
        raise ValueError(
            "validation scores are missing for: " + ", ".join(missing_scores)
        )
    best = sorted(
        candidates,
        key=lambda candidate: (validation_mse[candidate.run_id], candidate.run_id),
    )[:2]
    return [
        SearchCandidate(
            e_layers=candidate.e_layers,
            learning_rate=candidate.learning_rate,
            dropout=dropout,
            profile="tuned-dropout",
        )
        for candidate in best
        for dropout in (0.0, 0.2)
    ]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_best(
    manifests: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    if not manifests:
        raise ValueError("at least one validation manifest is required")
    records: list[dict[str, object]] = []
    for path in manifests:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid S4M manifest: {path}") from exc
        if record.get("status") != "validation_finalized":
            raise ValueError(f"manifest is not validation-finalized: {path}")
        if bool(record.get("test_evaluated", False)):
            raise ValueError(f"manifest has already accessed test data: {path}")
        checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"manifest checkpoint is missing: {checkpoint}")
        digest = sha256(checkpoint.read_bytes()).hexdigest()
        if digest != record.get("checkpoint_sha256"):
            raise ValueError(f"manifest checkpoint hash mismatch: {path}")
        record["manifest"] = str(path.resolve())
        records.append(record)
    records.sort(
        key=lambda record: (
            float(record["validation_mse"]),
            float(record["validation_mae"]),
            str(record["run_id"]),
        )
    )
    best = records[0]
    ranking = [
        {
            "run_id": record["run_id"],
            "validation_mse": record["validation_mse"],
            "validation_mae": record["validation_mae"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "manifest": record["manifest"],
        }
        for record in records
    ]
    selection: dict[str, object] = {
        "selected_run_id": best["run_id"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "selected_manifest": best["manifest"],
        "selection_metric": "validation_mse_then_mae",
        "test_metrics_seen": False,
        "candidates": ranking,
    }
    _atomic_json(output, selection)
    return selection


def build_train_command(
    candidate: SearchCandidate,
    *,
    python: str,
    s4m_root: Path,
    config: Path,
    artifacts: Path,
    output: Path,
    device: str,
    variant: str,
) -> list[str]:
    run_id = candidate.run_id_for(variant)
    return [
        python,
        (s4m_root.resolve() / "s4m" / "run.py").as_posix(),
        "--fair_protocol", "1",
        "--fair_mode", "train",
        "--fair_variant", variant,
        "--fair_profile", candidate.profile,
        "--fair_optimizer", "adam",
        "--fair_weight_decay", "0.0",
        "--fair_lradj", "none",
        "--fair_device", device,
        "--malr_config", config.resolve().as_posix(),
        "--malr_dataset", "ETTh1",
        "--malr_artifacts", artifacts.resolve().as_posix(),
        "--fair_run_dir", output.resolve().as_posix(),
        "--fair_run_id", run_id,
        "--fair_eval_reference_batch", "128",
        "--fair_upstream_commit", UPSTREAM_COMMIT,
        "--wandb_mode", "disabled",
        "--seed", "2021",
        "--model", "S4M",
        "--data", "custom4",
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", "96",
        "--e_layers", str(candidate.e_layers),
        "--d_model", "256",
        "--d_ff", "256",
        "--dropout", str(candidate.dropout),
        "--learning_rate", str(candidate.learning_rate),
        "--train_epochs", "15",
        "--patience", "5",
        "--batch_size", "16",
        "--num_workers", "0",
        "--per_mem_size", "5",
        "--en_conv_hidden_size", "256",
        "--topK", "20",
        "--thres1", "0.95",
        "--thres2", "0.6",
        "--M", "50",
        "--memory_size", "200",
        "--momentum", "0.99",
        "--short_len", "32",
        "--num_pattn", "30",
        "--num_dG", "1",
        "--encoder_type", "3",
        "--memnet", "13",
        "--mem_type", "3",
        "--itr", "1",
        "--mean_type", "global_mean",
        "--fillna", "mean",
    ]


def _replace_argument(command: list[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = value


def _run_commands(
    candidates: Iterable[SearchCandidate],
    *,
    args: argparse.Namespace,
    variant: str = "missing",
) -> list[Path]:
    manifests: list[Path] = []
    for candidate in candidates:
        command = build_train_command(
            candidate,
            python=sys.executable,
            s4m_root=args.s4m_root,
            config=args.config,
            artifacts=args.artifacts,
            output=args.output,
            device=args.device,
            variant=variant,
        )
        manifest = (
            args.output.resolve()
            / "runs"
            / f"{candidate.run_id_for(variant)}.json"
        )
        manifests.append(manifest)
        if manifest.exists():
            print(f"reuse finalized validation run: {manifest}")
            continue
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        completed = subprocess.run(
            command,
            cwd=args.s4m_root.resolve() / "s4m",
            env=build_s4m_subprocess_environment(args.s4m_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"S4M candidate failed ({completed.returncode}): "
                f"{candidate.run_id_for(variant)}"
            )
    return manifests


def _candidate_from_manifest(path: Path) -> SearchCandidate:
    record = json.loads(path.read_text(encoding="utf-8"))
    parameters = record["hyperparameters"]
    return SearchCandidate(
        e_layers=int(parameters["e_layers"]),
        learning_rate=float(parameters["learning_rate"]),
        dropout=float(parameters["dropout"]),
        profile=str(record["profile"]),
    )


def _evaluate_selected(args: argparse.Namespace) -> None:
    if args.selection is None:
        raise ValueError("--selection is required with --evaluate-selected")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    manifest_path = Path(str(selection["selected_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = _candidate_from_manifest(manifest_path)
    command = build_train_command(
        candidate,
        python=sys.executable,
        s4m_root=args.s4m_root,
        config=args.config,
        artifacts=args.artifacts,
        output=args.output,
        device=args.device,
        variant=str(manifest["variant"]),
    )
    _replace_argument(command, "--fair_mode", "evaluate")
    _replace_argument(command, "--fair_run_id", str(selection["selected_run_id"]))
    command.extend(
        [
            "--fair_selection",
            args.selection.resolve().as_posix(),
            "--fair_checkpoint",
            Path(str(selection["checkpoint"])).resolve().as_posix(),
        ]
    )
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return
    completed = subprocess.run(
        command,
        cwd=args.s4m_root.resolve() / "s4m",
        env=build_s4m_subprocess_environment(args.s4m_root),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"selected S4M evaluation failed: {completed.returncode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only S4M search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--s4m-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", choices=("official", "tuned"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-selected", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_modes = sum(
        (bool(args.profile), bool(args.smoke), bool(args.evaluate_selected))
    )
    if selected_modes != 1:
        raise ValueError(
            "choose exactly one of --profile, --smoke, or --evaluate-selected"
        )
    if args.evaluate_selected:
        _evaluate_selected(args)
        return 0
    if args.smoke:
        candidate = official_candidates()[0]
        command = build_train_command(
            candidate,
            python=sys.executable,
            s4m_root=args.s4m_root,
            config=args.config,
            artifacts=args.artifacts,
            output=args.output,
            device=args.device,
            variant="missing",
        )
        _replace_argument(command, "--fair_mode", "smoke")
        _replace_argument(command, "--d_model", "32")
        _replace_argument(command, "--d_ff", "32")
        _replace_argument(command, "--train_epochs", "1")
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            return 0
        completed = subprocess.run(
            command,
            cwd=args.s4m_root.resolve() / "s4m",
            env=build_s4m_subprocess_environment(args.s4m_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"S4M smoke test failed: {completed.returncode}")
        return 0

    if args.profile == "official":
        candidates = official_candidates()
        manifests = _run_commands(candidates, args=args)
        if args.dry_run:
            return 0
        selection_path = args.output.resolve() / "selection-official.json"
        selection = select_best(manifests, selection_path)
        best_manifest = Path(str(selection["selected_manifest"]))
        clean_candidate = _candidate_from_manifest(best_manifest)
        clean_manifests = _run_commands(
            [clean_candidate], args=args, variant="clean"
        )
        select_best(
            clean_manifests,
            args.output.resolve() / "selection-official-clean.json",
        )
        return 0

    stage1 = tuned_stage1_candidates()
    stage1_manifests = _run_commands(stage1, args=args)
    if args.dry_run:
        return 0
    stage1_scores = {
        str(json.loads(path.read_text(encoding="utf-8"))["run_id"]): float(
            json.loads(path.read_text(encoding="utf-8"))["validation_mse"]
        )
        for path in stage1_manifests
    }
    refined = dropout_refinement_candidates(stage1, stage1_scores)
    refined_manifests = _run_commands(refined, args=args)
    select_best(
        [*stage1_manifests, *refined_manifests],
        args.output.resolve() / "selection-tuned.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
