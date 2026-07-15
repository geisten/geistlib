# A home assistant that lives on your Raspberry Pi — and nowhere else

1.2 GB. One Raspberry Pi 5. No network hop, no account, no cloud. You ask it to
turn on the hallway light, and about two seconds later the light is on — and the
only computer that heard you is the one on your shelf.

That's geist-home: a single binary that runs the whole loop — speech
understanding, tool routing, device control — on the board. The model is baked
into the executable; the only things it ever talks to are your Home Assistant
instance on the LAN and the light you asked about. Most voice assistants ship
your words to someone else's data center and hope the service still exists next
year. This one makes the opposite bet.

Here's how it works — and the three ideas that make a 2-bit language model on an
$80 board responsive enough to actually use.

## Idea 1: the model routes, it never controls

The instinct with LLM agents is to let the model do everything — parse the
request, pick the device, decide the action, emit the API call. That's also how
you get a light that turns off because the model hallucinated an entity id.

geist-home draws a hard line. The model does exactly one probabilistic thing:
it picks **which tool** handles the request — a *command* tool (turn on/off,
dim, set temperature) or a *status* tool (read a device's state). Everything
after that is deterministic C:

- Which device? Matched against a registry file (`entity | domain | aliases`)
  by counting how many alias words appear in the request. "Licht im Wohnzimmer"
  scores 2 on `licht wohnzimmer`, beating the generic `licht` — no guessing.
- What action? Parsed from verbs and numbers in the request.
- Is it allowed? Writes are whitelisted to lights, switches, climate, covers.
  Garage doors and alarms are refused outright. Unlocking a door takes a
  two-turn confirmation flow — the model only ferries the confirm word, it
  never *decides* a security question.

The security promise isn't "we prompt-engineered the model to behave." It's
that the code to browse the web or read your files **isn't in the binary**. A
`make home` build compiles in the two home tools and nothing else. Fixed scope
at build time.

## Idea 2: the routing is calibrated, not vibes

Ask a small model "which tool?" and its raw preference is polluted by token
frequency — it likes tool names that happen to be common words. So geist-home
scores each tool name's first-token probability, then **subtracts a baseline**:
the same score for a content-free request. What's left (a pointwise mutual
information) is the request-driven signal.

On top of that sit deterministic evidence gates. A request with no home words
can't route to a home tool. A request that names a room but no specific device
("Wie ist der Status im Wohnzimmer?") gets the *whole room* listed back —
`light.wohnzimmer: an, cover.living_room_window: offen` — because a device is
"in" a room when its aliases carry the room word.

And a rule that costs nothing — no model pass at all: the read/write boundary is
decided by **sentence shape**, not the name score. An imperative ("Dimme das
Licht") means the command tool; a question ("Wie warm ist es?") means status.
First-token probabilities confuse the two constantly; the grammar of the
sentence doesn't.

## Idea 3: latency is a prefill problem, so stop re-paying it

Here's where it got interesting. The first measurements on the Pi showed a
12–13 second turn. Unacceptable for "turn on the light." The obvious suspect —
a 2-bit model decoding slowly — turned out to be wrong. Home answers are short
and mostly deterministic (`OK: light.flur → an`); the model barely decodes
anything. **The entire cost is prefill** — feeding the prompt through the
network before the first output token.

Three fixes, each measured:

1. **Pre-warm.** The 12–13 s was the *cold first turn* — priming the KV cache
   and computing the routing baseline. So the daemon runs one throwaway turn at
   startup, before it accepts requests. The first *real* request drops to ~3 s.

2. **A pinned routing menu.** Every turn re-fed the tool menu (~60 tokens)
   through the model just to route. That menu is constant, so we pin it into the
   KV cache once and rewind to it each turn — re-feeding only the ~8 tokens of
   the actual request. Routing prefill: 2.0 s → 0.45 s. (The subtle part: this
   required reordering the prompt so the constant menu is a prefix and the
   request is a suffix — and then proving, via the eval, that the reorder moved
   *zero* routing decisions.)

3. **A bounded context window.** The daemon keeps one long conversation so
   follow-ups work ("turn on the light" → "dim it down" resolves *it*). But an
   unbounded transcript meant per-turn prefill climbing toward *minutes* over a
   day of use. Capping the carry to the last couple of turns keeps it flat —
   and the follow-up context you actually need is only ever a turn or two back.

The result: a warm turn settled at **~2 seconds and holds there**, down from
"~4 s and climbing." The headline number was never about the model's speed. It
was about not re-paying for the same tokens on every turn.

## Why this shape is the right one

The reflex in 2026 is to throw a bigger model at the problem. But a home
appliance doesn't need a model that can do anything — it needs one that reliably
picks between "command" and "status" and lifts a device name. That's a job a
2-bit 2B model does well *when it's wrapped in deterministic scaffolding that
handles everything it shouldn't be trusted with.*

Small model, tight scope, everything measured, nothing leaves the house. The
interesting part was never making the model bigger. It was making it responsible
for less.

---

*geist-home is a `make home` build of geist, a from-scratch CPU inference engine
in C. Deployment guide: [HOME.md](../HOME.md).*
