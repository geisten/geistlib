"""A chunked campaign resumes without re-running or merging interrupted questions (#171)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1] / "tools"
with patch.object(sys, "path", [str(TOOLS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("eval_decision_campaign", TOOLS / "eval_decision_campaign.py")
    campaign = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(campaign)
ev = campaign.ev

# A stand-in for bench_decision: reads the wire input, emits one runtime line
# and every (mode, trial) sample per case; FAKE_STALL_AFTER stalls after that
# many cases so the deadline stops it mid-run.
FAKE_DRIVER = r'''#!/usr/bin/env python3
import json, math, os, sys, time
args = sys.argv[1:]
modes = args[args.index("--modes") + 1].split(",")
stall = int(os.environ.get("FAKE_STALL_AFTER", "-1"))
log = os.environ.get("FAKE_LOG")
print(json.dumps({"kind": "runtime", "backend": args[1], "vocab": 20, "kv_mode": "fp32", "version": "t"}), flush=True)
for case, line in enumerate(sys.stdin):
    if case == stall:
        time.sleep(60)
    ids = [int(x) for x in line.split()]
    np, nc = ids[:2]
    if log:
        open(log, "a").write(f"{modes[0]} {ids[2]}\n")
    logits = [2., 1., 0., -1.]
    total = sum(math.exp(x - 2.) for x in logits)
    probabilities = [math.exp(x - 2.) / total for x in logits]
    for trial in range(1 + int(args[5]) + int(args[6])):
        for order, mode in enumerate(modes):
            s = {"kind": "sample", "case_index": case, "mode": mode, "trial": trial,
                 "phase": "first" if trial == 0 else "warm", "order": order, "elapsed_ms": 10.,
                 "prompt_tokens": np, "candidate_count": nc, "process_peak_rss_bytes": 1000,
                 "generated_ids": []}
            if mode.startswith(("decision", "scorealt")):
                s.update(best_index=0, logits=logits, conditional_probabilities=probabilities,
                         projected_rows=4 if mode == "decision_selected" else 20,
                         logit_readback_bytes=16 if mode == "decision_selected" else 80,
                         head_ns=10 if mode == "decision_selected" else 0, generation_stop="not_generated")
            else:
                gen = [ids[2 + np]] if mode == "generate_1" else [3, 9]
                text = "A" if mode == "generate_1" else "x</think>A"
                s.update(generated_ids=gen, generation_stop="limit" if mode == "generate_1" else "eos",
                         generated_text_complete=True, generated_utf8_hex=text.encode().hex(),
                         surface_decode_ms=.01, best_index=None)
            print(json.dumps(s), flush=True)
'''


class CampaignTests(unittest.TestCase):
    def bundle(self, directory, n):
        profile_base = {"candidate_ids": [3, 4, 5, 6], "prompt_text": "Answer:",
                        "prompt_text_sha256": hashlib.sha256(b"Answer:").hexdigest(),
                        "prompt_build_ms": 1., "tokenize_ms": 2., "boundary_check_ms": 3.}
        lines = []
        for i in range(n):
            question = {"subject": "math" if i % 2 else "law", "question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": 0}
            profile = {**profile_base, "prompt_ids": [100 + i, 2]}
            lines.append(json.dumps({**question, "question_id": ev.data.question_key(question),
                                     "id": f"q{i}/rotation0", "target_index": 0, "split": "development",
                                     "variant": 0, "profiles": {p: profile for p in ev.PROFILES}}))
        (directory / "development.jsonl").write_text("\n".join(lines) + "\n")
        ev.write_json(directory / "split_audit.json", {"prepared_counts": {"test": 1}})
        (directory / "chat_template.jinja").write_text("template")
        ev.write_json(directory / "metadata.json", {
            "protocol": "geist-decision-mmlu-cases-v1", "rotations": 1, "per_subject": 1,
            "counts": {"development": {"questions": n, "records": n,
                                       "sha256": ev.bench.sha256(directory / "development.jsonl")}},
            "split_audit_sha256": ev.bench.sha256(directory / "split_audit.json"),
            "gguf_metadata": {"template_sha256": ev.bench.sha256(directory / "chat_template.jinja"), "eos_token_id": 9},
            "tokenizer_binary_sha256": "tokenizer", "prompt_policy": "test-only", "shots": 5,
            "template_date": "2026-10-04", "label_policy": "ABCD", "dataset": "synthetic",
            "dataset_revision": "0" * 40, "dataset_subset": True})

    def test_interrupted_chunk_resumes_without_rerunning_completed_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases"
            cases.mkdir()
            self.bundle(cases, 3)
            model, binary = root / "model", root / "driver"
            model.write_bytes(b"synthetic model")
            binary.write_text(FAKE_DRIVER)
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
            meta = ev.read_json(cases / "metadata.json")
            meta["model_sha256"] = ev.bench.sha256(model)
            ev.write_json(cases / "metadata.json", meta)
            log = root / "calls.log"
            args = SimpleNamespace(cases_dir=cases, split="development", purpose="pilot", model=model,
                                   binary=binary, backend="metal", threads=1, decode_cap=4, warmup=0,
                                   repeats=1, margin_pp=2., calibration=None, out_dir=root / "run",
                                   chunk_questions=2, deadline_minutes=.05, cool_max_c=65.,
                                   cool_timeout_minutes=1., status=False)
            gate = {"cool_gate": "temperature", "cool_wait_seconds": 0, "gpu_temp_c": 50., "cpu_temp_c": 50.}
            with (
                patch.object(campaign.bench_host, "wait_for_apple_cool", return_value=gate),
                patch.object(campaign.bench_host, "find_macmon", return_value=None),
                patch.object(ev.bench, "command_output", return_value="test"),
                patch.dict(os.environ, {"FAKE_LOG": str(log), "FAKE_STALL_AFTER": "1"}),
            ):
                # Chunk 0: questions 0 and 1; the first profile stalls after
                # question 0 and the deadline stops it.
                value = campaign.run(args)
                self.assertEqual(value["complete_by_profile"]["chat_direct"], 1)
                self.assertEqual(value["pending"], 3)
                with patch.dict(os.environ, {"FAKE_STALL_AFTER": "-1"}):
                    args.deadline_minutes = 1.
                    value = campaign.run(args)  # chunk 1: 0's other profiles, 1 completely
                    self.assertEqual(value["pending"], 1)
                    value = campaign.run(args)  # chunk 2: question 2, then finalize
                    self.assertEqual(value["pending"], 0)
                    self.assertEqual(value["chunks"], 3)
                # A changed evaluator input refuses to resume.
                args.decode_cap = 5
                with self.assertRaisesRegex(ValueError, "frozen manifest differs"):
                    campaign.run(args)
            calls = [line.split() for line in log.read_text().splitlines()]
            # The stalled question never started its samples; every profile ran each question once.
            # decision_dense leads chat_direct and cloze, generate_long the reasoning profile.
            self.assertEqual(sorted(int(p) - 100 for m, p in calls if m == "decision_dense"), [0, 0, 1, 1, 2, 2])
            self.assertEqual(sorted(int(p) - 100 for m, p in calls if m == "generate_long"), [0, 1, 2])
            stalled = ev.read_json(args.out_dir / "chunks/0000/chat_direct.run.json")
            self.assertTrue(stalled["stopped_at_deadline"])
            report = ev.read_json(args.out_dir / "report.json")
            self.assertEqual(report["profiles"]["chat_reasoning"]["modes"]["generate_long"]["quality"]["n"], 3)
            self.assertEqual(report["campaign"]["chunks"], 3)
            self.assertEqual(report["campaign"]["interrupted_driver_runs"], 1)
            self.assertIn("chat_direct/decision_selected", report["latency_by_start_temperature"])


    def setup_run(self, root, n):
        cases = root / "cases"
        cases.mkdir()
        self.bundle(cases, n)
        model, binary = root / "model", root / "driver"
        model.write_bytes(b"synthetic model")
        binary.write_text(FAKE_DRIVER)
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        meta = ev.read_json(cases / "metadata.json")
        meta["model_sha256"] = ev.bench.sha256(model)
        ev.write_json(cases / "metadata.json", meta)
        return SimpleNamespace(cases_dir=cases, split="development", purpose="pilot", model=model,
                               binary=binary, backend="metal", threads=1, decode_cap=4, warmup=0,
                               repeats=1, margin_pp=2., calibration=None, out_dir=root / "run",
                               chunk_questions=10, deadline_minutes=1., cool_max_c=65.,
                               cool_timeout_minutes=1., status=False, calibration_modes=None, import_from=None,
                               core_arms=False, spotcheck=0, spotcheck_seed="s", question_ids=None,
                               reasoning_only=False)

    def test_import_reuses_complete_samples_without_driver_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.setup_run(root, 4)
            log = root / "calls.log"
            gate = {"cool_gate": "temperature", "cool_wait_seconds": 0, "gpu_temp_c": 50., "cpu_temp_c": 50.}
            with (
                patch.object(campaign.bench_host, "wait_for_apple_cool", return_value=gate),
                patch.object(campaign.bench_host, "find_macmon", return_value=None),
                patch.object(ev.bench, "command_output", return_value="test"),
                patch.dict(os.environ, {"FAKE_LOG": str(log), "FAKE_STALL_AFTER": "-1"}),
            ):
                args.chunk_questions = 2
                campaign.run(args)  # questions 0 and 1 complete in the first run
                calls = len(log.read_text().splitlines())
                first = args.out_dir
                args.out_dir, args.import_from, args.chunk_questions = root / "second", first, 10
                value = campaign.run(args)  # imports 0 and 1, runs only 2 and 3
                self.assertEqual(value["pending"], 0)
                new = [int(line.split()[1]) - 100 for line in log.read_text().splitlines()[calls:]]
                self.assertEqual(sorted(set(new)), [2, 3])
                self.assertNotIn(0, new)
                self.assertNotIn(1, new)
                report = ev.read_json(args.out_dir / "report.json")
                self.assertEqual(report["profiles"]["chat_direct"]["modes"]["decision_dense"]["quality"]["n"], 4)
                args.out_dir, args.decode_cap = root / "third", 5
                with self.assertRaisesRegex(ValueError, "import: decode_cap"):
                    campaign.run(args)
            with self.assertRaisesRegex(ValueError, "calibration split only"):
                args.calibration_modes = ["decision_dense"]
                args.out_dir = root / "fourth"
                ev.prepare_run(args)

    def test_derived_sample_is_a_valid_declared_bundle(self):
        sampler = importlib.util.spec_from_file_location("sample_decision_cases", TOOLS / "sample_decision_cases.py")
        with patch.object(sys, "path", [str(TOOLS), *sys.path]):
            module = importlib.util.module_from_spec(sampler)
            sampler.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root, 10)
            module.derive(root, root / "sample", "development", 4, "seed")
            meta, rows = ev.load_cases(root / "sample", "development", "pilot")
            self.assertEqual(len(rows), 4)
            self.assertEqual(meta["development_sample"]["allocation"], {"law": 2, "math": 2})
            self.assertEqual(meta["development_sample"]["population"], 10)
            with self.assertRaises(FileExistsError):
                module.derive(root, root / "sample", "development", 4, "seed")


    def test_single_trial_core_arms_with_determinism_spotcheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.setup_run(root, 6)
            args.repeats, args.core_arms, args.spotcheck = 0, True, 2
            args.spotcheck_seed = "seed"
            log = root / "calls.log"
            gate = {"cool_gate": "temperature", "cool_wait_seconds": 0, "gpu_temp_c": 50., "cpu_temp_c": 50.}
            with (
                patch.object(campaign.bench_host, "wait_for_apple_cool", return_value=gate),
                patch.object(campaign.bench_host, "find_macmon", return_value=None),
                patch.object(ev.bench, "command_output", return_value="test"),
                patch.dict(os.environ, {"FAKE_LOG": str(log), "FAKE_STALL_AFTER": "-1"}),
            ):
                value = campaign.run(args)  # all 6 questions, one trial each
                self.assertEqual((value["complete"], value["spotcheck_pending"]), (6, 2))
                self.assertFalse((args.out_dir / "report.json").exists())
                value = campaign.run(args)  # the spot-check chunk, then the report
                self.assertEqual(value["pending"], 0)
            modes = {line.split()[0] for line in log.read_text().splitlines()}
            self.assertEqual(modes, {"decision_dense", "generate_long"})  # no cloze, no reference arms
            report = ev.read_json(args.out_dir / "report.json")
            self.assertEqual(set(report["profiles"]), {"chat_direct", "chat_reasoning"})
            self.assertEqual(set(report["profiles"]["chat_direct"]["modes"]), {"decision_dense", "decision_selected"})
            spot = report["determinism_spotcheck"]
            self.assertTrue(spot["deterministic"])
            self.assertEqual(spot["by_arm"]["chat_reasoning/generate_long"], {"questions": 2, "identical": 2})
            self.assertEqual(len(spot["declared"]["question_ids"]), 2)
            self.assertEqual(report["model_query_count_including_reference_arms_and_repeats"], 18)


    def test_reasoning_only_rerun_on_frozen_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.setup_run(root, 5)
            rows = [json.loads(l) for l in (args.cases_dir / "development.jsonl").read_text().splitlines()]
            ids = root / "ids.json"
            ids.write_text(json.dumps([rows[3]["question_id"], rows[1]["question_id"]]))
            args.question_ids, args.reasoning_only, args.decode_cap = ids, True, 8
            log = root / "calls.log"
            gate = {"cool_gate": "temperature", "cool_wait_seconds": 0, "gpu_temp_c": 50., "cpu_temp_c": 50.}
            with (
                patch.object(campaign.bench_host, "wait_for_apple_cool", return_value=gate),
                patch.object(campaign.bench_host, "find_macmon", return_value=None),
                patch.object(ev.bench, "command_output", return_value="test"),
                patch.dict(os.environ, {"FAKE_LOG": str(log), "FAKE_STALL_AFTER": "-1"}),
            ):
                value = campaign.run(args)
            self.assertEqual(value["pending"], 0)
            calls = [line.split() for line in log.read_text().splitlines()]
            self.assertEqual(calls, [["generate_long", "103"], ["generate_long", "101"]])  # frozen order, one profile
            report = ev.read_json(args.out_dir / "report.json")
            self.assertEqual(set(report["profiles"]), {"chat_reasoning"})
            self.assertNotIn("comparison", report)
            self.assertIn("frozen list of 2 of 5", report["population"])
            ids.write_text(json.dumps(["not-a-question"]))
            args.out_dir = root / "bad"
            with self.assertRaisesRegex(ValueError, "question-ids"):
                ev.prepare_run(args)


if __name__ == "__main__":
    unittest.main()
