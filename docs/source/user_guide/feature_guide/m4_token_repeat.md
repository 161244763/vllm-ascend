# M4 token repetition detection

The default `token_repeat.algorithm` remains `"window"`. Select `"m4"` to use
fixed M4 rules with 64 raw-token check boundaries and a final partial-block flush.
M4 requires default values for the legacy window-scoring parameters.

Build the CPU extension independently in the serving environment (Linux, C++17):

```bash
pip install ./tools/m4_token_repeat/src
python -m m4_vocab --tokenizer /model/tokenizer.json --out /model/m4.npz
```

The table generator supports ByteLevel BPE only. Deploy the NPZ and tokenizer.json
for the served tokenizer together; their SHA-256 fingerprints are checked against
the model tokenizer. Remote model IDs require tokenizer.json in the local HF cache.

Set these fields under `detector.token_repeat` in runtime_config:

```json
{
  "enabled": true,
  "algorithm": "m4",
  "m4_tables": "/model/m4.npz",
  "m4_tokenizer_json": "/model/tokenizer.json",
  "m4_think_end": -1
}
```

Use `-1` for chat mode, or the model's explicit `</think>` token ID for reasoning.
Drain requests before changing resources; use new file paths rather than replacing
cached tables in place. Missing resources or extension errors are logged through
runtime_guard's existing soft-failure path; inference may continue without detection.

M4 uses the existing CPU detection queue and Incident actions. Terminal tails are
checked before Store cleanup. Disable clears private state; disabling all detectors
also clears output IO while retaining prompts. Failed native batches are quarantined
and never replayed. Focused regression tests are in `tests/ut/runtime_guard/test_m4_production.py`.
