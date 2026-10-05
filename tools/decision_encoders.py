"""Optional, local-only reference encoders; not native geist C backends.

Importing this module never imports torch, transformers or laya. The factory
requires an explicit feature opt-in. Inputs which the native tokenizer/builder
would shorten are rejected before inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import decision_metrics as metrics


class InputRejected(ValueError):
    """The complete request cannot be represented by this backend."""


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact_manifest(directory):
    """Hash weights, configuration and tokenizers, excluding download cache."""
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*") if p.is_file()
                   and ".cache" not in p.relative_to(directory).parts
                   and p.suffix in (".json", ".safetensors"))
    if not paths or not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError("a local checkpoint with safetensors is required")
    return {str(p.relative_to(directory)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in paths}


@dataclass(frozen=True)
class ChoiceRequest:
    state: str
    instructions: str
    options: dict[str, str]

    def validate(self):
        if (not isinstance(self.state, str) or not self.state.strip()
                or not isinstance(self.instructions, str) or not self.instructions.strip()
                or not isinstance(self.options, dict) or not 2 <= len(self.options) <= 255
                or any(not isinstance(k, str) or not k.strip()
                       or not isinstance(v, str) or not v.strip() for k, v in self.options.items())):
            raise ValueError("nonempty state/instructions and 2..255 named descriptions required")


def probabilities(logits, temperature=1.0):
    return [math.exp(x) for x in metrics.log_probabilities(logits, temperature)]


def validate_scores(logits, count):
    if (len(logits) != count or any(type(x) not in (int, float) or not math.isfinite(x)
                                  for x in logits)):
        raise ValueError("backend returned incomplete or nonfinite scores")


def validate_literal_text(tokenizer, texts):
    # Tokenizers interpret these strings as controls. Do not silently erase
    # a literal [MASK] in user data or count it as the answer position.
    controls = set(tokenizer.all_special_tokens)
    if any(control and control in text for text in texts for control in controls):
        raise InputRejected("input contains a reserved tokenizer control string")


class TorchBackend:
    def __init__(self, device, dtype):
        import torch
        if device not in ("cpu", "mps") or dtype not in ("float32", "float16"):
            raise ValueError("explicit cpu/mps device and float32/float16 dtype required")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no automatic CPU fallback")
        self.torch, self.device = torch, torch.device(device)
        self.dtype = getattr(torch, dtype)

    def synchronize(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def _forward(self, fn):
        self.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = fn()
        self.synchronize()
        return output, (time.perf_counter() - start) * 1000

    def close(self):
        self.model = None
        if self.device.type == "mps":
            self.torch.mps.empty_cache()


class ModernBertBackend(TorchBackend):
    name = "modernbert"
    capabilities = ("choice",)

    def __init__(self, directory, device, dtype, max_tokens):
        super().__init__(device, dtype)
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        directory = Path(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForMaskedLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, attn_implementation="sdpa", reference_compile=False)
        self.model.eval().to(self.device)
        native = self.model.config.max_position_embeddings
        if type(max_tokens) is not int or not 0 < max_tokens <= native:
            raise ValueError("token cap must fit the checkpoint's native context")
        self.max_tokens = max_tokens
        self.temperature = 1.0
        self.metadata = {"adapter": self.name, "device": device, "dtype": dtype,
                         "attention": "sdpa", "reference_compile": False,
                         "max_tokens": max_tokens, "native_context": native,
                         "capabilities": list(self.capabilities),
                         "calibration": "none; candidate-conditional softmax",
                         "prompt_policy": "modernbert-instruct-mask-choice-v1"}

    def prepare(self, request):
        request.validate()
        tok = self.tokenizer
        validate_literal_text(tok, [request.state, request.instructions, *request.options,
                                    *request.options.values()])
        labels = [tok.encode(k, add_special_tokens=False) for k in request.options]
        if any(len(ids) != 1 for ids in labels) or len({ids[0] for ids in labels}) != len(labels):
            raise InputRejected("ModernBERT answer keys must be distinct single tokens")
        text = (request.instructions + "\n" + request.state + "\nCHOICES:\n"
                + "\n".join(f"- {k}: {v}" for k, v in request.options.items())
                + "\nANSWER: [unused0] " + tok.mask_token)
        encoded = tok(text, return_tensors="pt", truncation=False)
        ids = encoded["input_ids"][0].tolist()
        masks = [i for i, token in enumerate(ids) if token == tok.mask_token_id]
        if len(ids) > self.max_tokens:
            raise InputRejected(f"complete prompt has {len(ids)} tokens; cap {self.max_tokens}")
        if len(masks) != 1:
            raise InputRejected("exactly one answer mask required")
        return {"encoded": encoded, "input_ids": ids, "mask_positions": masks,
                "candidate_ids": [v[0] for v in labels], "prompt_text": text,
                "keys": list(request.options)}

    def score(self, request):
        start = time.perf_counter()
        prepared = self.prepare(request)
        inputs = {k: v.to(self.device) for k, v in prepared["encoded"].items()}
        self.synchronize()
        prepare_ms = (time.perf_counter() - start) * 1000
        def forward():
            # Evaluate the vocabulary head only at the answer position. This
            # is mathematically the same head as AutoModelForMaskedLM.forward;
            # no input tokens or backbone layers are skipped.
            hidden = self.model.model(**inputs).last_hidden_state
            hidden = hidden[:, prepared["mask_positions"][0]:prepared["mask_positions"][0] + 1]
            return self.model.decoder(self.model.head(hidden))[0, 0]
        output, forward_ms = self._forward(forward)
        post = time.perf_counter()
        logits = output[prepared["candidate_ids"]].float().cpu().tolist()
        validate_scores(logits, len(prepared["keys"]))
        p = probabilities(logits)
        selected = max(range(len(p)), key=p.__getitem__)
        result = {"choice": prepared["keys"][selected], "logits": logits, "probabilities": p,
                  "temperature": 1.0, "input_ids": prepared["input_ids"],
                  "mask_positions": prepared["mask_positions"],
                  "candidate_ids": prepared["candidate_ids"], "prompt_text": prepared["prompt_text"],
                  "prepare_ms": prepare_ms, "forward_ms": forward_ms}
        result["postprocess_ms"] = (time.perf_counter() - post) * 1000
        result["request_ms"] = (time.perf_counter() - start) * 1000
        return result


class LayaBackend(TorchBackend):
    name = "laya"
    capabilities = ("choice", "score", "noul")

    def __init__(self, directory, device, dtype, max_tokens):
        super().__init__(device, dtype)
        from laya import common
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        directory = Path(directory)
        self.common = common
        self.cfg = read_json(directory / "rl_agent_config.json")
        self.tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True,
                                                       trust_remote_code=False)
        native = self.cfg["max_len"]
        if type(max_tokens) is not int or not 0 < max_tokens <= native:
            raise ValueError("token cap must fit the checkpoint's trained context")
        self.max_tokens = max_tokens
        self.model = common.build_model(self.cfg, encoder_dir=str(directory / "encoder"), pretrained=False)
        self.model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.metadata = {"adapter": self.name, "device": device, "dtype": dtype,
                         "attention": "sdpa", "compile": False,
                         "max_tokens": max_tokens, "native_context": native,
                         "head_max_tokens": self.cfg["head_max_len"], "capabilities": list(self.capabilities),
                         "prompt_policy": "laya-native-lossless-choice-v1",
                         "calibration": "checkpoint temperatures with SDK bounds; not fitted or validated on MMLU",
                         "sdk_policy": "native builder/model; no SDK fallback or input truncation"}

    @staticmethod
    def normalize_question(question):
        if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
            raise ValueError("typed choice/score/noul question required")
        ins = question.get("instructions")
        if not isinstance(ins, str) or not ins.strip():
            raise ValueError("nonempty question instructions required")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            ChoiceRequest("validation", ins, criteria).validate()
        elif kind == "score":
            if (not isinstance(criteria, list) or not 2 <= len(criteria) <= 50
                    or any(not isinstance(v, str) or not v.strip() for v in criteria)):
                raise ValueError("score requires 2..50 textual levels")
        elif criteria is not None and (not isinstance(criteria, dict)
                or set(criteria) != {"false", "true"}
                or any(not isinstance(v, str) or not v.strip() for v in criteria.values())):
            raise ValueError("noul criteria must describe false and true")
        return {"t": kind, "ins": ins, "crit": criteria}

    def prepare_questions(self, state, questions):
        if not isinstance(state, str) or not state.strip() or not isinstance(questions, dict) or not questions:
            raise ValueError("nonempty textual state and typed question map required")
        tok, common = self.tokenizer, self.common
        prepared = []
        for key, question in questions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("nonempty question key required")
            q = self.normalize_question(question)
            options = common.render_options(q)
            validate_literal_text(tok, [state, q["ins"], *options])
            encode = lambda text: tok.encode(text, add_special_tokens=False)
            expected = [tok.cls_token_id] + encode(q["t"] + " question: " + q["ins"]) + [tok.sep_token_id]
            markers = []
            for option in options:
                markers.append(len(expected))
                expected += [tok.mask_token_id] + encode(" " + option)
            expected += [tok.sep_token_id] + encode(state) + [tok.sep_token_id]
            ids, actual_markers, option_stats, state_stats = common.build_sequence(
                tok, state, q, self.max_tokens, self.cfg["head_max_len"],
                return_stats=True, return_truncation_stats=True)
            if ids != expected or actual_markers != markers or state_stats["truncated"]:
                raise InputRejected("Laya native builder would shorten instructions, options or state")
            prepared.append({"key": key, "question": q, "ids": ids, "markers": markers,
                             "qtype": common.QTYPES[q["t"]], "options": option_stats,
                             "state_stats": state_stats})
        return prepared

    def decide(self, state, questions):
        start = time.perf_counter()
        items = self.prepare_questions(state, questions)
        batch = self.common.collate_items([items], self.tokenizer.pad_token_id)
        batch = {k: batch[k].to(self.device) for k in
                 ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
        self.synchronize()
        prepare_ms = (time.perf_counter() - start) * 1000
        outputs, forward_ms = self._forward(lambda: self.model(**batch))
        post = time.perf_counter()
        logits_all = outputs[0].float().cpu().tolist()
        if len(logits_all) != len(items):
            raise ValueError("backend returned incomplete question rows")
        answers = {}
        for item, raw in zip(items, logits_all):
            q = item["question"]
            logits = raw[:len(item["markers"])]
            validate_scores(logits, len(item["markers"]))
            temperature = self.cfg.get("temperature_by_options", {}).get(
                self.common.temp_bucket(item["qtype"], len(logits)), self.cfg["temperature"][item["qtype"]])
            temperature = self.common.clamp_temperature(temperature)
            p = probabilities(logits, temperature)
            answer = {"type": q["t"], "logits": logits, "probabilities": p, "temperature": temperature,
                      "answer_confidence": max(p), "input_ids": item["ids"],
                      "mask_positions": item["markers"], "question": q, "state": state}
            if q["t"] == "choice":
                answer["choice"] = list(q["crit"])[max(range(len(p)), key=p.__getitem__)]
            elif q["t"] == "score":
                answer["score"] = sum(i * value for i, value in enumerate(p))
            else:
                answer["noul"] = p[1]
            answers[item["key"]] = answer
        result = {"answers": answers, "prepare_ms": prepare_ms, "forward_ms": forward_ms,
                  "postprocess_ms": (time.perf_counter() - post) * 1000}
        result["request_ms"] = (time.perf_counter() - start) * 1000
        return result

    def score(self, request):
        request.validate()
        result = self.decide(request.state, {"answer": {"type": "choice", "instructions": request.instructions,
                                                       "criteria": request.options}})
        return {**result["answers"]["answer"], **{k: v for k, v in result.items() if k != "answers"}}


def create_backend(name, directory, *, enabled=False, device="cpu", dtype="float32", max_tokens=1024):
    if enabled is not True:
        raise RuntimeError("encoder reference backends disabled; explicit feature opt-in required")
    classes = {"modernbert": ModernBertBackend, "laya": LayaBackend}
    if name not in classes:
        raise ValueError("unknown encoder backend")
    # In particular, Laya's builder accepts Hub IDs as well as directories.
    # Reject incomplete local layouts before importing its optional SDK.
    directory = Path(directory).expanduser().resolve(strict=True)
    config_path = "config.json" if name == "modernbert" else "encoder/config.json"
    required = ("model.safetensors", config_path, "tokenizer.json", "tokenizer_config.json")
    if name == "laya":
        required = ("model.safetensors", config_path, "rl_agent_config.json",
                    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
    if not directory.is_dir() or any(not (directory / p).is_file() for p in required):
        raise ValueError("complete local checkpoint, encoder config and tokenizer required")
    if read_json(directory / config_path).get("model_type") != "modernbert":
        raise ValueError("these reference adapters require a ModernBERT encoder")
    return classes[name](directory, device, dtype, max_tokens)
