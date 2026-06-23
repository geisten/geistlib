#!/usr/bin/env python3
"""eval_tooling.py — function-calling & JSON-generation quality for a geist GGUF.

Self-contained: drives the `eval_geist` REPL (the `GEN` command greedily
generates and detokenizes with the model's own tokenizer), so there is no
external dependency. It measures the practical "structured output" capability
that powers tool use:

  * json     — given a request, produce a valid JSON object matching a schema.
  * func     — given a tool spec + a user query, produce a JSON function call
               with the right name and arguments (BFCL-style, format-agnostic:
               the JSON is extracted from the reply, whether the model wraps it
               in ```json fences, native tool tokens, or emits it bare).

Both suites are curated probe sets (embedded below) — small, deterministic, and
runnable with no dataset download. Each task ships its own validator, so the
score reflects real correctness (valid JSON + right function + right args), not
string overlap.

Usage:
  make bench-tooling
  python3 tools/eval_tooling.py --bin bin/<target>/release/tools/eval_geist \
      --gguf model.gguf [--suite json|func|all] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Gemma 4 chat template (turn markers <|turn> = 105, <turn|> = 106).
TURN = "<bos><|turn>user\n{u}<turn|>\n<|turn>model\n"


class Repl:
    def __init__(self, binary: str, gguf: str):
        self.proc = subprocess.Popen(
            [binary, gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                sys.exit(f"eval_geist exited before READY.\n{err}")
            if line.strip() == "READY":
                break

    def gen(self, prompt: str, max_new: int = 160) -> str:
        esc = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(f"GEN {max_new} {esc}\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().rstrip("\n")
        if not resp.startswith("OK "):
            sys.exit(f"eval_geist GEN error: {resp}")
        body = resp[3:]
        return body.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.terminate()


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (```json fences, bare
    braces, or after prose). Returns the parsed object or None."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    # Greedy first-{...}-balanced fallback.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        break
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ----------------------------------------------------------------------------
# JSON-generation probe set: (prompt, validator(obj) -> bool)
# ----------------------------------------------------------------------------
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


JSON_TASKS = [
    ("Return a JSON object with keys \"name\" (string) and \"age\" (number) for "
     "a person named Alice who is 30. Output only JSON.",
     lambda o: o.get("name") == "Alice" and _is_num(o.get("age")) and o["age"] == 30),
    ("Output a JSON object describing the book '1984' by George Orwell published "
     "in 1949, with keys title, author, year. JSON only.",
     lambda o: "1984" in str(o.get("title", "")) and "Orwell" in str(o.get("author", ""))
               and _is_num(o.get("year")) and o["year"] == 1949),
    ("Extract structured data as JSON with keys name, age, occupation from: "
     "\"John Smith is a 42-year-old engineer.\" JSON only.",
     lambda o: "John" in str(o.get("name", "")) and o.get("age") == 42
               and "engineer" in str(o.get("occupation", "")).lower()),
    ("Return JSON {\"result\": <number>} for the sum of 128 and 256. JSON only.",
     lambda o: _is_num(o.get("result")) and o["result"] == 384),
    ("Return a JSON object with key \"is_prime\" (boolean): is 17 a prime number? "
     "JSON only.",
     lambda o: o.get("is_prime") is True),
    ("Return JSON with a key \"colors\" whose value is a list of the three primary "
     "colors of light. JSON only.",
     lambda o: isinstance(o.get("colors"), list) and len(o["colors"]) == 3),
    ("Return JSON {\"celsius\": <number>} converting 212 degrees Fahrenheit to "
     "Celsius. JSON only.",
     lambda o: _is_num(o.get("celsius")) and abs(o["celsius"] - 100) < 1.0),
    ("Output a JSON object with keys \"city\" and \"country\" for the capital of "
     "Japan. JSON only.",
     lambda o: "Tokyo" in str(o.get("city", "")) and "Japan" in str(o.get("country", ""))),
    ("Return a JSON object with keys \"a\", \"b\", \"c\" mapped to 1, 2, 3 "
     "respectively. JSON only.",
     lambda o: o.get("a") == 1 and o.get("b") == 2 and o.get("c") == 3),
    ("Return JSON {\"even\": [...]} listing the even numbers from 1 to 10. JSON only.",
     lambda o: isinstance(o.get("even"), list) and sorted(o["even"]) == [2, 4, 6, 8, 10]),
    # --- harder: nesting, lists of objects, strict types, multiple constraints ---
    ("Return JSON for a user named Bob with a nested \"address\" object containing "
     "city 'Paris' and country 'France'. Keys: name, address{city,country}. JSON only.",
     lambda o: o.get("name") == "Bob" and isinstance(o.get("address"), dict)
               and o["address"].get("city") == "Paris"
               and o["address"].get("country") == "France"),
    ("Return JSON {\"items\": [...]} with two items, each with \"id\" (number) and "
     "\"label\" (string): id 1 label 'a', id 2 label 'b'. JSON only.",
     lambda o: isinstance(o.get("items"), list) and len(o["items"]) == 2
               and o["items"][0].get("id") == 1 and o["items"][0].get("label") == "a"
               and o["items"][1].get("id") == 2 and o["items"][1].get("label") == "b"),
    ("Return JSON {\"price\": <number>, \"in_stock\": <boolean>} for a product "
     "priced 19.99 that is in stock. JSON only.",
     lambda o: _is_num(o.get("price")) and abs(o["price"] - 19.99) < 0.01
               and o.get("in_stock") is True),
    ("Return JSON with keys \"username\" (the lowercase string \"alice\"), \"score\" "
     "(the integer 95), and \"passed\" (boolean true). JSON only.",
     lambda o: o.get("username") == "alice" and o.get("score") == 95
               and isinstance(o.get("score"), int) and o.get("passed") is True),
]


# ----------------------------------------------------------------------------
# Function-calling probe set: (tools_desc, query, validator(call) -> bool)
# The model is asked to reply ONLY with {"name": ..., "arguments": {...}}.
# ----------------------------------------------------------------------------
def _args(o):
    return o.get("arguments") or o.get("args") or o.get("parameters") or {}


FUNC_TOOLS_SINGLE = 'get_weather(city: string) — current weather for a city'
FUNC_TASKS = [
    (FUNC_TOOLS_SINGLE, "What is the weather in Tokyo?",
     lambda c: c.get("name") == "get_weather"
               and "tokyo" in str(_args(c).get("city", "")).lower()),
    ('calculate(expression: string) — evaluate a math expression',
     "What is 15 times 23?",
     lambda c: c.get("name") == "calculate"
               and "15" in str(_args(c)) and "23" in str(_args(c))),
    ('set_timer(minutes: number) — start a countdown timer',
     "Set a timer for 10 minutes.",
     lambda c: c.get("name") == "set_timer" and _num_eq(_args(c).get("minutes"), 10)),
    ('search_web(query: string) — search the web',
     "Find Italian restaurants near me.",
     lambda c: c.get("name") == "search_web"
               and "italian" in str(_args(c).get("query", "")).lower()),
    ('send_email(to: string, subject: string, body: string) — send an email',
     "Email alice@example.com with subject 'Hi' and body 'Hello there'.",
     lambda c: c.get("name") == "send_email"
               and "alice@example.com" in str(_args(c).get("to", ""))),
    ('translate(text: string, target_lang: string) — translate text',
     "Translate 'good morning' to French.",
     lambda c: c.get("name") == "translate"
               and "french" in str(_args(c).get("target_lang", "")).lower()),
    # Tool selection among several:
    ('get_weather(city: string); get_time(city: string); get_news(topic: string)',
     "What time is it in New York?",
     lambda c: c.get("name") == "get_time"
               and "new york" in str(_args(c).get("city", "")).lower()),
    ('get_weather(city: string); get_time(city: string); get_news(topic: string)',
     "Give me the latest news about technology.",
     lambda c: c.get("name") == "get_news"
               and "tech" in str(_args(c).get("topic", "")).lower()),
    ('play_music(song: string, artist: string) — play a song',
     "Play Bohemian Rhapsody by Queen.",
     lambda c: c.get("name") == "play_music"
               and "queen" in str(_args(c).get("artist", "")).lower()),
    ('convert_currency(amount: number, from_currency: string, to_currency: string)',
     "Convert 100 US dollars to euros.",
     lambda c: c.get("name") == "convert_currency"
               and _num_eq(_args(c).get("amount"), 100)),
    # --- harder: all-args-correct, numeric exactness, distractor, nested/list args ---
    ('send_email(to: string, subject: string, body: string) — send an email',
     "Email bob@test.com with the subject 'Meeting' and the body 'See you at 3pm'.",
     lambda c: c.get("name") == "send_email"
               and "bob@test.com" in str(_args(c).get("to", ""))
               and "meeting" in str(_args(c).get("subject", "")).lower()
               and "3pm" in str(_args(c).get("body", "")).lower().replace(" ", "")),
    ('book_flight(origin: string, destination: string, passengers: number)',
     "Book a flight from London to Paris for 3 passengers.",
     lambda c: c.get("name") == "book_flight"
               and "london" in str(_args(c).get("origin", "")).lower()
               and "paris" in str(_args(c).get("destination", "")).lower()
               and _num_eq(_args(c).get("passengers"), 3)),
    ('get_current_weather(city: string); get_forecast(city: string, days: number)',
     "What is the 5-day forecast for Berlin?",
     lambda c: c.get("name") == "get_forecast"
               and "berlin" in str(_args(c).get("city", "")).lower()
               and _num_eq(_args(c).get("days"), 5)),
    ('create_event(title: string, date: string, attendees: array of string)',
     "Create an event titled 'Lunch' on 2025-06-15 with attendees Alice and Bob.",
     lambda c: c.get("name") == "create_event"
               and "lunch" in str(_args(c).get("title", "")).lower()
               and isinstance(_args(c).get("attendees"), list)
               and any("alice" in str(a).lower() for a in _args(c)["attendees"])
               and any("bob" in str(a).lower() for a in _args(c)["attendees"])),
]


def _num_eq(v, target):
    try:
        return abs(float(v) - target) < 1e-6
    except (TypeError, ValueError):
        return False


def run_json(repl: Repl, verbose: bool):
    valid = schema = 0
    for i, (prompt, check) in enumerate(JSON_TASKS):
        reply = repl.gen(TURN.format(u=prompt))
        obj = extract_json(reply)
        v = obj is not None
        s = v and _safe(check, obj)
        valid += v
        schema += s
        if verbose:
            print(f"  [json {i+1}/{len(JSON_TASKS)}] valid={v} schema={s}  {reply[:70]!r}")
    n = len(JSON_TASKS)
    print(f"JSON generation:  valid JSON {valid}/{n} ({valid/n:.0%}), "
          f"schema-correct {schema}/{n} ({schema/n:.0%})")
    return valid, schema, n


def run_func(repl: Repl, verbose: bool):
    valid = name_ok = full = 0
    instr = ("\nReply ONLY with a JSON object of the form "
             '{"name": "<function>", "arguments": {...}} and nothing else.')
    for i, (tools, query, check) in enumerate(FUNC_TASKS):
        prompt = f"You can call these functions:\n{tools}\n\nUser: {query}{instr}"
        reply = repl.gen(TURN.format(u=prompt))
        call = extract_json(reply)
        v = call is not None
        nm = v and "name" in call
        ok = v and _safe(check, call)
        valid += v
        name_ok += nm
        full += ok
        if verbose:
            print(f"  [func {i+1}/{len(FUNC_TASKS)}] valid={v} correct={ok}  {reply[:70]!r}")
    n = len(FUNC_TASKS)
    print(f"Function calling: valid JSON {valid}/{n} ({valid/n:.0%}), "
          f"named call {name_ok}/{n} ({name_ok/n:.0%}), "
          f"fully correct {full}/{n} ({full/n:.0%})")
    return valid, full, n


def _safe(check, obj):
    try:
        return bool(check(obj))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suite", choices=["json", "func", "all"], default="all")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-correct", type=float, default=0.0,
                    help="gate: exit non-zero if (json schema-correct + func "
                         "fully-correct) / total < this fraction (0 = report only)")
    args = ap.parse_args()
    if not Path(args.bin).is_file():
        sys.exit(f"eval_tooling: binary not found: {args.bin} (run `make` first)")

    repl = Repl(args.bin, args.gguf)
    correct = total = 0
    try:
        print("== geist tooling benchmark (Gemma 4 chat template, greedy) ==")
        if args.suite in ("json", "all"):
            _v, schema, n = run_json(repl, args.verbose)
            correct += schema
            total += n
        if args.suite in ("func", "all"):
            _v, full, n = run_func(repl, args.verbose)
            correct += full
            total += n
    finally:
        repl.close()

    if args.min_correct > 0.0:
        rate = correct / total if total else 0.0
        print(f"tooling: {correct}/{total} correct ({rate:.0%}), floor {args.min_correct:.0%}")
        if rate < args.min_correct:
            sys.exit(f"TOOLING REGRESSION: {rate:.0%} < {args.min_correct:.0%}")


if __name__ == "__main__":
    main()
