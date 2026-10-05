"""Capped/invalid baselines and exploratory pilots cannot qualify a speed claim."""
import hashlib
import importlib.util
import json
import math
import subprocess
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("eval_decision", TOOLS / "eval_decision.py")
eval_tool = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    SPEC.loader.exec_module(eval_tool)


class DecisionEvaluationTests(unittest.TestCase):
    def sample(self, text="thinking</think>\nA", ids=None, stop="eos"):
        return {"generated_ids": [1, 2, 9] if ids is None else ids, "generation_stop": stop,
                "generated_text_complete": True, "generated_utf8_hex": text.encode().hex(),
                "surface_decode_ms": .01}

    def test_final_answer_and_stop_are_separate(self):
        case = {"target_index": 0}
        result = eval_tool.generation_outcome(self.sample(), case, True, 9, 10, 20)
        self.assertTrue(result["correct"])
        self.assertFalse(result["truncated"])
        result = eval_tool.generation_outcome(self.sample(ids=[1, 2, 3], stop="limit"), case, True, 9, 3, 20)
        self.assertTrue(result["correct"])
        self.assertTrue(result["truncated"])
        for text in ("thinking about A", "thinking</think>A or B"):
            result = eval_tool.generation_outcome(self.sample(text), case, True, 9, 10, 20)
            self.assertFalse(result["valid"])
            self.assertFalse(result["correct"])
        broken = {**self.sample(), "generated_utf8_hex": "ff"}
        self.assertEqual(eval_tool.generation_outcome(broken, case, True, 9, 10, 20)["invalid_reason"], "invalid_utf8")
        broken = {**self.sample(), "generation_stop": "limit"}
        with self.assertRaisesRegex(ValueError, "stop"):
            eval_tool.generation_outcome(broken, case, True, 9, 10, 20)

    def test_quality_matched_gate_is_not_a_latency_only_win(self):
        outcomes = {f"q{i}": True for i in range(1000)}
        timing = [{"question_id": k, "inference_p50_ms": 1., "additive_component_estimate_ms": 2.} for k in outcomes]
        baseline = [{**r, "inference_p50_ms": 10., "additive_component_estimate_ms": 12.} for r in timing]
        quality = {"invalid_answers": 0, "capped_outputs": 0}
        result = eval_tool.comparison(outcomes, outcomes, timing, baseline, "quality", 2., quality)
        self.assertTrue(result["quality_matched_speedup_eligible"])
        self.assertEqual(result["paired_quality"]["inference_p50_ms_ratio_reasoning_over_selected"], 10.)
        for purpose, q in (("pilot", quality), ("quality", {**quality, "capped_outputs": 1}),
                           ("quality", {**quality, "invalid_answers": 1})):
            self.assertFalse(eval_tool.comparison(outcomes, outcomes, timing, baseline, purpose, 2., q)["quality_matched_speedup_eligible"])
        with self.assertRaisesRegex(ValueError, "populations differ"):
            eval_tool.comparison(outcomes, outcomes, timing[:-1], baseline[:-1], "quality", 2., quality)

    def bundle(self, directory, split="test", per_subject=0):
        question = {"subject": "math", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 0}
        profile = {"prompt_ids": [1, 2], "candidate_ids": [3, 4, 5, 6], "prompt_text": "Answer:",
                   "prompt_text_sha256": hashlib.sha256(b"Answer:").hexdigest(),
                   "prompt_build_ms": 1., "tokenize_ms": 2., "boundary_check_ms": 3.}
        row = {**question, "question_id": eval_tool.data.question_key(question), "id": "q/rotation0",
               "target_index": 0, "split": split, "variant": 0,
               "profiles": {p: profile for p in eval_tool.PROFILES}}
        (directory / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        eval_tool.write_json(directory / "split_audit.json", {"prepared_counts": {"test": 1}})
        (directory / "chat_template.jinja").write_text("template")
        meta = {"protocol": "geist-decision-mmlu-cases-v1", "rotations": 1, "per_subject": per_subject,
                "counts": {split: {"questions": 1, "records": 1, "sha256": eval_tool.bench.sha256(directory / f"{split}.jsonl")}},
                "split_audit_sha256": eval_tool.bench.sha256(directory / "split_audit.json"),
                "gguf_metadata": {"template_sha256": eval_tool.bench.sha256(directory / "chat_template.jinja")}}
        eval_tool.write_json(directory / "metadata.json", meta)
        return row

    def test_integrity_and_held_out_scope_cannot_be_silently_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.bundle(directory)
            self.assertEqual(len(eval_tool.load_cases(directory, "test", "quality")[1]), 1)
            with self.assertRaisesRegex(ValueError, "development"):
                eval_tool.load_cases(directory, "test", "pilot")
            self.bundle(directory)
            meta = eval_tool.read_json(directory / "metadata.json")
            meta["subject_limit"] = 1
            eval_tool.write_json(directory / "metadata.json", meta)
            with self.assertRaisesRegex(ValueError, "complete"):
                eval_tool.load_cases(directory, "test", "quality")
            self.bundle(directory, per_subject=1)
            with self.assertRaisesRegex(ValueError, "complete"):
                eval_tool.load_cases(directory, "test", "quality")
            (directory / "test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                eval_tool.load_cases(directory, "test", "quality")

    def test_orchestration_freezes_manifest_and_retains_raw_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases"
            cases.mkdir()
            self.bundle(cases, split="development", per_subject=1)
            model, binary = root / "model", root / "binary"
            model.write_bytes(b"synthetic model identity")
            binary.write_bytes(b"synthetic driver identity")
            meta = eval_tool.read_json(cases / "metadata.json")
            meta.update(model_sha256=eval_tool.bench.sha256(model), tokenizer_binary_sha256="tokenizer",
                        prompt_policy="test-only", shots=5, template_date="2026-10-04", label_policy="ABCD",
                        dataset="synthetic-test-fixture", dataset_revision="0" * 40, dataset_subset=True)
            meta["gguf_metadata"]["eos_token_id"] = 9
            eval_tool.write_json(cases / "metadata.json", meta)
            args = SimpleNamespace(cases_dir=cases, split="development", purpose="pilot", model=model,
                                   binary=binary, backend="cpu_neon", threads=1, decode_cap=4,
                                   warmup=0, repeats=1, margin_pp=2., calibration=None, out_dir=root / "run")
            def driver(command, *, input, env, stdout, stderr, text):
                self.assertTrue((args.out_dir / "manifest.json").exists())
                modes = command[command.index("--modes") + 1].split(",")
                ids = [int(x) for x in input.split()]
                np, nc = ids[:2]
                for record in [{"kind": "runtime", "backend": args.backend, "vocab": 20, "kv_mode": "fp32"}]:
                    stdout.write(json.dumps(record) + "\n")
                logits = [2., 1., 0., -1.]
                total = sum(math.exp(x - 2.) for x in logits)
                probabilities = [math.exp(x - 2.) / total for x in logits]
                for trial in range(2):
                    for order, mode in enumerate(modes):
                        sample = {"kind": "sample", "case_index": 0, "mode": mode,
                                  "trial": trial, "phase": "first" if trial == 0 else "warm", "order": order,
                                  "elapsed_ms": 10., "prompt_tokens": np, "candidate_count": nc,
                                  "process_peak_rss_bytes": 1000, "generated_ids": []}
                        if mode in eval_tool.NUMERIC:
                            sample.update(best_index=0, logits=logits, conditional_probabilities=probabilities,
                                          projected_rows=4 if mode == "decision_selected" else 20,
                                          logit_readback_bytes=16 if mode == "decision_selected" else 80,
                                          head_ns=10 if mode == "decision_selected" else 0,
                                          generation_stop="not_generated")
                        else:
                            sample.update(self.sample("A" if mode == "generate_1" else "x</think>A",
                                                      [ids[2 + np]] if mode == "generate_1" else [3, 9],
                                                      "limit" if mode == "generate_1" else "eos"))
                            sample["best_index"] = None
                        stdout.write(json.dumps(sample) + "\n")
                return subprocess.CompletedProcess(command, 0)
            with (
                patch.object(eval_tool.platform, "system", return_value="Linux"),
                patch.object(eval_tool.platform, "processor", return_value="test-cpu"),
                patch.object(eval_tool.platform, "platform", return_value="test-os"),
                patch.object(eval_tool.bench, "command_output", return_value="test"),
                patch.object(eval_tool.subprocess, "run", side_effect=driver),
            ):
                eval_tool.run(args)
            report = eval_tool.read_json(args.out_dir / "report.json")
            self.assertEqual(report["profiles"]["chat_reasoning"]["modes"]["generate_long"]["quality"]["accuracy"], 1.)
            self.assertFalse(report["comparison"]["quality_matched_speedup_eligible"])
            self.assertTrue((args.out_dir / "chat_direct.driver.jsonl").exists())
            with self.assertRaises(FileExistsError):
                eval_tool.run(args)
            args.out_dir = root / "failed"
            def failed(command, **kwargs):
                kwargs["stdout"].write("partial output\n")
                return subprocess.CompletedProcess(command, 1)
            with (
                patch.object(eval_tool.platform, "system", return_value="Linux"),
                patch.object(eval_tool.platform, "processor", return_value="test-cpu"),
                patch.object(eval_tool.platform, "platform", return_value="test-os"),
                patch.object(eval_tool.bench, "command_output", return_value="test"),
                patch.object(eval_tool.subprocess, "run", side_effect=failed),
            ):
                with self.assertRaisesRegex(RuntimeError, "raw partial outputs retained"):
                    eval_tool.run(args)
            self.assertTrue((args.out_dir / "manifest.json").exists())
            self.assertEqual((args.out_dir / "chat_direct.driver.jsonl").read_text(), "partial output\n")


if __name__ == "__main__":
    unittest.main()
