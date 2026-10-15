"""Builds the #48 phase-0 corpora: profile (expert usage is measured on it) and
held-out (the hot set is checked against it). Usage: corpus.py <out> <geistlib> <llama.cpp>.
Prose from Project Gutenberg (public domain), code from the two checkouts; every
file's sha256 goes to <out>/corpus.sha256."""
import glob, hashlib, os, sys, urllib.request

out, G, L = sys.argv[1:4]
BOOKS = {"en_pride": 1342, "en_moby": 2701, "de_verwandlung": 22367, "de_effi": 5323}


def book(name):
    p = f"{out}/src/{name}.txt"
    if not os.path.exists(p):
        os.makedirs(f"{out}/src", exist_ok=True)
        n = BOOKS[name]
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt", p)
    return open(p, encoding="utf-8", errors="replace").read()


def chunks(text, n, size=13000, skip=20000):
    text = text[skip:]
    step = max(size, (len(text) - size) // n)
    return [text[i * step:i * step + size] for i in range(n)]


def code(pattern, n):
    return chunks("".join(open(p, errors="replace").read() for p in sorted(glob.glob(pattern))), n, skip=0)


sets = {
    "profile": [("en", chunks(book("en_pride"), 4)), ("de", chunks(book("de_verwandlung"), 3, skip=3000)),
                ("c", code(f"{G}/src/engine/*.c", 4)), ("py", code(f"{G}/tools/*.py", 2))],
    "heldout": [("en", chunks(book("en_moby"), 4)), ("de", chunks(book("de_effi"), 3)),
                ("c", code(f"{L}/src/llama-*.cpp", 4)), ("py", code(f"{L}/gguf-py/gguf/*.py", 2))],
}
with open(f"{out}/corpus.sha256", "w") as h:
    for s, groups in sets.items():
        os.makedirs(f"{out}/{s}", exist_ok=True)
        for g, cs in groups:
            for i, c in enumerate(cs):
                p = f"{out}/{s}/{g}{i}.txt"
                open(p, "w").write(c)
                h.write(hashlib.sha256(c.encode()).hexdigest() + "  " + p.split("/", 1)[-1] + "\n")
