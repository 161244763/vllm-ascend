#!/usr/bin/env python3
"""Build tokenizer-specific M4 lookup tables for ByteLevel BPE.

Columns encode normalized IDs, letter flags, character/whitespace counts,
repeated characters, reasoning markers and sentence endings. Tokenizer
fingerprints are embedded in the generated NPZ for deployment validation.
"""

import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SEP = -1
TABLES = os.path.join(HERE, "tables")
# '</think>' token id of the corpus's generation mode: the answer starts there. DeepSeek ran without a
# reasoning block (chat mode): its '</think>' tokens are answer text, not a boundary -> None.
THINK_END = {"qwen3": 151668, "glm": 154842, "dsv4": None}
# words a no-progress reasoning loop keeps returning to (normalised form: lower-case, alphanumeric only); chosen before
# calibration from the v3.5 rule's description of such loops ("Hmm ...", "Wait, ...") and their Chinese counterparts
MARKERS = {
    "wait",
    "hmm",
    "alternatively",
    "actually",
    "maybe",
    "perhaps",
    "but",
    "however",
    "reconsider",
    "again",
    "no",
    "ok",
    "okay",
    "let",
    "不过",
    "或者",
    "等等",
    "嗯",
    "但是",
    "可能",
    "再",
    "重新",
    "还是",
    "对吗",
}


def _byte_decoder():
    """inverse of GPT-2 bytes_to_unicode (byte-level BPE alphabet -> byte)"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def build(tokenizer_json):
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(tokenizer_json)
    V = tk.get_vocab_size()
    with open(tokenizer_json, encoding="utf-8") as source:
        definition = json.load(source)
        special = {a["id"] for a in definition.get("added_tokens", []) if a.get("special")}

    def has_byte_level(node):
        if isinstance(node, dict):
            return node.get("type") == "ByteLevel" or any(has_byte_level(v) for v in node.values())
        if isinstance(node, list):
            return any(has_byte_level(v) for v in node)
        return False

    if definition.get("model", {}).get("type") != "BPE" or not has_byte_level(definition.get("decoder")):
        raise ValueError("M4 table builder supports ByteLevel BPE tokenizer.json only")
    bd = _byte_decoder()
    norm = np.full(V, SEP, np.int32)
    letter = np.zeros(V, bool)
    chars = np.zeros(V, np.int16)
    ws = np.zeros(V, np.int16)
    run_char = np.full(V, -1, np.int32)
    marker = np.zeros(V, bool)
    send = np.zeros(V, bool)
    ids = {}  # normalised string -> id (0 unused)
    for i in range(V):
        piece = tk.id_to_token(i)
        if piece is None:
            continue
        if i in special:
            norm[i] = ids.setdefault(("special", i), len(ids) + 1)
            chars[i] = len(piece)
            continue
        try:
            s = bytes(bd[c] for c in piece).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            norm[i] = ids.setdefault(("bytes", piece), len(ids) + 1)  # partial UTF-8: opaque, same bytes = same id
            letter[i] = True
            chars[i] = 1
            continue
        chars[i] = min(len(s), 32767)
        send[i] = any(c in ".?!。？！\n" for c in s)
        ws[i] = min(sum(c.isspace() for c in s), 32767)
        if s and not s[0].isspace() and s == s[0] * len(s):
            run_char[i] = ord(s[0])
        key = "".join(c for c in s.lower() if c.isalnum())
        if key:
            norm[i] = ids.setdefault(key, len(ids) + 1)
            letter[i] = any(c.isalpha() for c in key)
            marker[i] = key in MARKERS
    return {
        "norm": norm,
        "letter": letter,
        "chars": chars,
        "ws": ws,
        "run_char": run_char,
        "marker": marker,
        "send": send,
    }


TABLE_CACHE_SIZE = 8


@lru_cache(maxsize=TABLE_CACHE_SIZE)
def load(corpus):
    """tables of a shipped tokenizer (qwen3 | glm | dsv4) or of a table file built by main() (a path ending in .npz)"""
    # Load on detector creation, never per token. Eviction releases only the
    # cache reference; live detectors keep their own arrays. Treat table files
    # as immutable for a deployment; cache_clear() is for a drained teardown.
    path = corpus if corpus.endswith(".npz") else os.path.join(TABLES, f"{corpus}.npz")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def main():
    ap = argparse.ArgumentParser(description="build the lookup table of one byte-level BPE tokenizer")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the served model")
    ap.add_argument("--out", required=True, help="output .npz")
    a = ap.parse_args()
    t = build(a.tokenizer)
    with open(a.tokenizer, "rb") as source:
        t["tokenizer_sha256"] = np.array(hashlib.sha256(source.read()).hexdigest())
    np.savez_compressed(a.out, **t)
    n = t["norm"]
    print(
        f"{a.out}: vocab {n.size}, separators {int((n == SEP).sum())}, "
        f"distinct normalised ids {len(np.unique(n[n > 0]))}, "
        f"run_char tokens {int((t['run_char'] >= 0).sum())}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
