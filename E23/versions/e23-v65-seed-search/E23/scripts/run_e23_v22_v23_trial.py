"""Run matched width-128 E23 control and eight-head trials on locked ETTh1 P4."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments"
V12_MAE = 0.5882120694703256
V12_MSE = 0.7454477286392869


def _run(args: list[str]) -> None:
    print("running", " ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def _common(config: Path, output: Path) -> list[str]:
    return [
        "--protocol-config", str(ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml"),
        "--e23-config", str(config),
        "--migration-catalog", str(ROOT.parent / "E22/migration/paper_catalog.jsonl"),
        "--mask-artifacts", str(ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"),
        "--output", str(output), "--device", "cuda:0",
    ]


def main() -> None:
    results = {}
    for label, config_name in (
        ("v22_singlehead_128", "e23_multi_expert_p4_width128_window_singlehead_v22.yaml"),
        ("v23_eighthead_128", "e23_multi_expert_p4_width128_window_eighthead_v23.yaml"),
    ):
        config = ROOT / "configs" / config_name
        trial = TRIALS / f"e23_{label}"
        stage1 = trial / "stage1"
        stage2 = trial / "stage2"
        stage1_manifest = stage1 / "runs/e23-stage1.json"
        stage2_manifest = stage2 / "runs/e23-stage2.json"
        _run([sys.executable, "-m", "integrations.e23.run", "train-repair",
              *_common(config, stage1)])
        _run([sys.executable, "-m", "integrations.e23.run", "train-forecast",
              *_common(config, stage2), "--stage1-manifest", str(stage1_manifest),
              "--selection-metric", "mae", "--forecast-loss", "mse_plus_0p2_mae"])
        replay = trial / "validation_replay.json"
        _run([sys.executable, str(ROOT / "scripts/replay_e23_p4_validation.py"),
              "--e23-config", str(config), "--stage2-manifest", str(stage2_manifest),
              "--output", str(replay), "--device", "cuda:0"])
        record = json.loads(stage2_manifest.read_text())
        results[label] = {
            "config": str(config), "stage2_manifest": str(stage2_manifest),
            "validation_mae": record["best_val_mae"],
            "validation_mse": record["best_val_mse"],
            "gate_passed": record["best_val_mae"] < V12_MAE and record["best_val_mse"] <= V12_MSE,
            "replay": str(replay),
        }
        print("selected validation", label, results[label], flush=True)
    passing = [name for name, result in results.items() if result["gate_passed"]]
    selected = min(passing, key=lambda name: (results[name]["validation_mae"],
                                              results[name]["validation_mse"])) if passing else "v12"
    if selected != "v12":
        result = results[selected]
        test_path = TRIALS / f"e23_{selected}" / "selected_test.json"
        _run([sys.executable, "-m", "integrations.e23.run", "evaluate",
              *_common(Path(result["config"]), test_path),
              "--selection", result["stage2_manifest"]])
        result["test"] = str(test_path)
    summary = {
        "protocol": "ETTh1 P4 fixed masks, seed 2021, 100+100 epochs",
        "v12_validation_mae": V12_MAE, "v12_validation_mse": V12_MSE,
        "selected": selected, "trials": results,
    }
    path = TRIALS / "e23_v22_v23_validation_selection.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
