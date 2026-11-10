# M4 token repetition detection

M4 is an explicit alternative to the existing single-window detector. The default
remains `algorithm: "window"`; existing window configurations retain their meaning.
M4 uses frozen CONFIG_M4 components, 64 raw-token blocks and the original tail
flush. It does not accept non-default window scoring knobs.

## Install and configure

Build the CPU extension separately in the serving Python environment (Linux,
C++17 compiler required); rebuilding vllm-ascend is unnecessary:

```bash
pip install ./tools/m4_token_repeat/src
python -m v35_detector.vocab --tokenizer /model/tokenizer.json --out /model/m4.npz
```

The builder supports ByteLevel BPE only and embeds the tokenizer.json SHA-256.
Do not reuse a Qwen table for another tokenizer. Deploy the generated NPZ and
its tokenizer.json together, as immutable files. The detector checks the source
fingerprint against both the table and the served model tokenizer.json. Remote
model IDs require an already-populated local Hugging Face cache; detection never
downloads tokenizer resources. Missing resources, unsupported builds and mismatched
fingerprints produce explicit runtime_guard errors; there is no algorithm fallback.
Runtime guard retains its existing observational soft-failure behavior, so monitor
those errors: inference continuing does not mean repetition detection succeeded.

Set these fields in the existing runtime configuration:

```json
{
  "detector": {
    "token_repeat": {
      "enabled": true,
      "algorithm": "m4",
      "m4_tables": "/model/m4.npz",
      "m4_tokenizer_json": "/model/tokenizer.json",
      "m4_think_end": -1
    }
  }
}
```

Use `-1` for chat mode, or the served tokenizer's explicit `</think>` token ID
for reasoning mode. No model-specific default is inferred. To change tables,
change the resource path/configuration with requests drained; replacing files
in place is unsupported because reference vocabulary loading uses a bounded cache.

## Code walkthrough and review checklist

- Algorithm: `tools/m4_token_repeat/src/m4_native.cpp` and `position_queue.h`
  are unchanged. `m4_cpp.py` keeps the original component reasons, block boundary,
  first-alert position and finish result. Review its integer bounds, table owners,
  state terminal flags and partial-consumption receipts before native deployment.
- Call chain: after_sample appends Store output once. The existing CPU job calls
  DetectorManager, then TokenRepeatDetector. M4 gathers unread suffixes via
  `_consumed_len`, calls `feed_events` once per batch and maps sparse events to
  Incidents with `fired_at`, `reason`, and `finish_only`.
- Finalization: the processor waits for reapability and drained CPU work, consumes
  any unread suffix, flushes tails, routes Incidents while Store IO still exists,
  then rechecks reapability because an alert can arm a dump job. A forced reap
  with outstanding CPU work remains cleanup-only. `clear_finished` itself never
  calls native finish. Scheduler-finished IDs use this terminal path; explicit
  cancellation through Store.clear is cleanup-only.
- State: detector operations/config refresh share an RLock. Config epochs invalidate
  old queued tasks after disable or algorithm/resource changes. Store identity checks
  reject tasks for reaped/reused IDs; completion counters also check that identity.
  Store removes shared identity before clearing private state. Review engine-level
  callback ordering on target, especially async scheduling and forced reap.
- Disable: includes PR #21346 head ec17eea's on_disabled hooks and prompt-preserving
  output purge. Native states, failed-generation markers and cursors are cleared.
  Tables retained by the reference loader's eight-entry cache are bounded; a cleared
  detector no longer owns native state. Re-enable creates fresh detection state.
- Failure: native batches are not atomic. All submitted generations are quarantined
  after failure; committed first alerts are recovered from receipts, and an explicit
  exception is logged. Neither the batch nor its remaining suffix is retried.
  Cleanup/disable removes quarantine markers. Inspect error logs as validation data.
- Tests: `tests/ut/runtime_guard/test_m4_production.py` covers batch/cursors, finish,
  disable/re-enable, cancellation cleanup, ID reuse, partial failures, resource errors
  and repeated cleanup. Synthetic differential tests compare first position/reason
  and finish semantics to the original Python M4. Native differential requires the
  independently installed extension; a skip is not a pass.

## Validation before production

Run the affected runtime_guard and runtime_config suites in the supported environment:

```bash
pytest tests/ut/runtime_guard tests/ut/runtime_config --cov=vllm_ascend.observability.runtime_guard --cov-branch
bash format.sh ci
```

Also run the native input/exception/resource checks shipped under
`tools/m4_token_repeat`, sanitizer builds, tokenizer-specific historical replay,
and a long repeated-enable/cancel/reuse memory soak. Synthetic inputs and mocked
native state establish adapter behavior only. They do not establish native memory
safety, tokenizer coverage, complete callback ordering or NPU performance.

For inference performance use one environment/model/workload, comparing token_repeat
off, original Python M4 and optimized M4. Record TPOT mean, p95 of per-request mean
TPOT, TTFT, throughput and total degradation relative to off. CPU microbenchmarks
are not substitutes. These hardware/replay/sanitizer runs have not been performed
in the Windows development environment.

Release note wording for the eventual PR (assign its real number when opened):
"Added an explicit M4 backend to runtime token repetition detection with batched CPU
execution and tokenizer-specific resource validation."
