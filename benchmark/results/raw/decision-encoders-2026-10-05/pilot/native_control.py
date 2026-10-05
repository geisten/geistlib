"""Recorded control: optimized MLM mask head and Laya SDK wire answers."""
from pathlib import Path
import json
import shutil
import sys
import torch

sys.path.insert(0, "/Users/germar/.codex/worktrees/decision-inference/geistlib/tools")
import decision_encoders as e
import eval_decision_encoders as runner

torch.set_num_threads(6)
root = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (root / "cases/development.jsonl").read_text().splitlines()]
report = {"scope": "Development controls, not accuracy/timing acceptance", "device": "mps",
          "dtype": "float32", "modernbert": [], "laya": []}
backend = e.create_backend("modernbert", Path("/private/tmp/decision-encoder-models/modernbert"),
                           enabled=True, device="mps", dtype="float32", max_tokens=1024)
for shots in (0, 5):
    for row in rows:
        request = runner.mmlu_request(row, shots)
        scored = backend.score(request)
        prepared = backend.prepare(request)
        inputs = {k: v.to(backend.device) for k, v in prepared["encoded"].items()}
        with torch.inference_mode():
            native = backend.model(**inputs).logits[0, prepared["mask_positions"][0],
                                                   prepared["candidate_ids"]].float().cpu().tolist()
        drift = max(abs(a - b) for a, b in zip(native, scored["logits"]))
        equal = max(range(4), key=native.__getitem__) == "ABCD".index(scored["choice"])
        report["modernbert"].append({"question_id": row["question_id"], "shots": shots,
                                     "native_logits": native, "mask_only_logits": scored["logits"],
                                     "candidate_max_abs_error": drift, "argmax_equal": equal})
        if drift > 1e-4 or not equal:
            raise RuntimeError("mask-only/native head parity control failed")
backend.close()
print("ModernBERT native candidate parity passed", flush=True)

source = Path("/private/tmp/decision-encoder-models/laya/typed-decisions")
clone = root / "sdk-checkpoint"
clone.mkdir(exist_ok=True)
for path in source.rglob("*"):
    if not path.is_file() or ".cache" in path.parts:
        continue
    target = clone / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        if not target.exists():
            target.hardlink_to(path)
    else:
        shutil.copyfile(path, target)
import laya
from laya.agent import Agent
native = Agent(str(clone), device="mps", fast=False, compile=False)
backend = e.create_backend("laya", source, enabled=True, device="mps", dtype="float32", max_tokens=1024)
for row in rows:
    request = runner.mmlu_request(row, 0)
    questions = {"answer": {"type": "choice", "instructions": request.instructions, "criteria": request.options}}
    scored = backend.decide(request.state, questions)["answers"]["answer"]
    expected = native.predict(request.state, questions)["answers"]["answer"]
    error = max(abs(a - expected["probabilities"][key]) for key, a in zip("ABCD", scored["probabilities"]))
    equal = scored["choice"] == expected["choice"]
    report["laya"].append({"question_id": row["question_id"], "sdk_answer": expected,
                           "adapter_choice": scored["choice"], "adapter_probabilities": scored["probabilities"],
                           "probability_max_abs_error_vs_rounded_sdk": error, "choice_equal": equal})
    if error > 5.1e-5 or not equal:
        raise RuntimeError("Laya SDK/reference adapter parity failed")
report["laya_sdk_cpu_fallback_count"] = native.cpu_fallback_count
if native.cpu_fallback_count:
    raise RuntimeError("unexpected native SDK CPU fallback")
backend.close()
(root / "native_control.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print("Laya native SDK choice/probability parity passed", flush=True)
