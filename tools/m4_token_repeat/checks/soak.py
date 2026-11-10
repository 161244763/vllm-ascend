"""Bounded CPU lifecycle soak. Admission toggles, never mid-request state loss.

Run against an unchanged release extension; no model, torch, or NPU imports.
RSS is process RSS (allocator retention included), not evidence of leak absence.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/vendor/m4/src')]
from perf_session import CPUBackend, PerfSession

ACTIVE = ('states', 'rows', 'received', 'computed_tokens', 'finish_only',
          'events', 'incarnations', 'closed', 'retired_rows', 'completed')


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
            k, v, _ = line.split()
            values[k.rstrip(':') + '_kib'] = int(v)
    values['fds'] = len(list(Path('/proc/self/fd').iterdir()))
    return values


def empty(session):
    for name in ACTIVE:
        assert not getattr(session, name), (name, len(getattr(session, name)))
    assert len(session.retired_lengths) <= session.max_generated_tokens + 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert args.seconds >= 5
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261109)
    random_ids = [rng.randrange(1000, 100000) for _ in range(8192)]
    backend = CPUBackend({})
    on = PerfSession({}, factory=lambda _: backend)
    off = PerfSession(None, true_off=True)
    sessions = (off, on)
    started = time.monotonic()
    counters = dict(cohorts=0, requests=0, on_requests=0, off_requests=0,
                    normal_finish=0, cancelled=0, zero_length=0, id_reuses=0,
                    admission_toggles=0, tokens=0, after_alert_tokens=0,
                    detector_rebuilds=0, stale_callbacks_checked=0)
    samples = []
    initial = memory()

    # Controlled live-state high-water checkpoints. Release every reference;
    # no malloc_trim, which would hide allocator retention from the soak.
    with (args.out / 'memory-checkpoints.jsonl').open('x') as checkpoint:
        for count in (1, 16, 64, 128):
            gc.collect()
            before = memory()
            keys = list(range(count))
            gen = on.register(keys)
            on.sampled(keys, [random_ids] * count, gen)
            live = memory()
            non_alert_states = sum(r['alert_at'] is None for r in on.rows.values())
            on.finish(keys, gen)
            on.finished(keys, gen)
            empty(on)
            gc.collect()
            checkpoint.write(json.dumps(dict(concurrency=count, tokens_per_request=8192,
                non_alert_states=non_alert_states, before=before, live=live,
                after_drain=memory())) + '\n')
            checkpoint.flush()

    # New sessions keep admission accounting separate from checkpoints.
    on = PerfSession({}, factory=lambda _: backend)
    off = PerfSession(None, true_off=True)
    sessions = (off, on)
    soak_start = time.monotonic()
    previous = {}
    previous_mode = None
    with (args.out / 'samples.jsonl').open('x') as stream:
        while time.monotonic() - soak_start < args.seconds:
            cycle = counters['cohorts']
            count = (1, 16, 64, 128)[cycle % 4]
            length = (0, 65, 128, 4096, 8192)[(cycle // 4) % 5]
            chunk = 1 if length == 128 and cycle % 2 == 0 else 65
            periodic = cycle % 3 == 0
            ids = [64] * length if periodic else random_ids[:length]
            # Alternating admission mode; existing requests remain routed to
            # their original session even when the next admission toggles.
            admitted = []
            for slot in range(count):
                mode = (cycle + slot) % 2
                session = sessions[mode]
                key = slot  # intentionally reused IDs, with explicit generations
                generation = session.register([key])[0]
                old = previous.get((mode, key))
                if old is not None:
                    before = session.inspect([key])
                    session.finish([key], [old])
                    session.finished([key], [old])
                    assert before == session.inspect([key])
                    counters['stale_callbacks_checked'] += 1
                    counters['id_reuses'] += 1
                previous[mode, key] = generation
                admitted.append((mode, key, generation))
                counters['admission_toggles'] += previous_mode is not None and previous_mode != mode
                previous_mode = mode
                counters['on_requests' if mode else 'off_requests'] += 1
            for start in range(0, length, chunk):
                part = ids[start:start + chunk]
                for mode in (0, 1):
                    subset = [(key, gen) for m, key, gen in admitted if m == mode]
                    if subset:
                        keys, gens = map(list, zip(*subset))
                        sessions[mode].sampled(keys, [part] * len(keys), gens)
            for mode, key, gen in admitted:
                session = sessions[mode]
                row = session.inspect([key])[key]
                assert row['generated'] == length
                assert 0 <= row['computed_tokens'] <= row['received'] <= length
                if row['alert_at'] is not None:
                    counters['after_alert_tokens'] += length - row['scored']
                cancel = (cycle + key) % 5 == 0
                if not cancel:
                    session.finish([key], [gen])
                    session.finish([key], [gen])  # idempotent end-of-stream
                counters['cancelled' if cancel else 'normal_finish'] += 1
                counters['zero_length'] += length == 0
                session.finished([key], [gen])
                session.finished([key], [gen])
                counters['requests'] += 1
                counters['tokens'] += length
            for session in sessions:
                empty(session)
            counters['cohorts'] += 1
            if counters['cohorts'] % 32 == 0:
                # Destroy native table owners only after all requests drain.
                # Preserve aggregate counters in the existing sessions.
                on.detector = None
                backend = None
                gc.collect()
                backend = CPUBackend({})
                on.detector = backend
                counters['detector_rebuilds'] += 1
            if not samples or time.monotonic() - soak_start - samples[-1]['elapsed_s'] >= 5:
                gc.collect()
                item = dict(elapsed_s=time.monotonic() - soak_start,
                            **memory(), **counters,
                            live_states=sum(len(s.states) for s in sessions),
                            length_bins=[len(s.retired_lengths) for s in sessions])
                stream.write(json.dumps(item) + '\n')
                stream.flush()
                samples.append(item)
                print(json.dumps(item), flush=True)
    assert sum(s.retired_requests for s in sessions) == counters['requests']
    assert sum(s.retired_generated for s in sessions) == counters['tokens']
    for session in sessions:
        empty(session)
    gc.collect()
    tail = [s for s in samples if s['elapsed_s'] >= args.seconds / 2]
    xs = [s['elapsed_s'] for s in tail]
    ys = [s['VmRSS_kib'] for s in tail]
    slope = None
    if len(xs) > 1:
        xbar, ybar = sum(xs)/len(xs), sum(ys)/len(ys)
        slope = sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)
    summary = dict(complete=True, pid=os.getpid(), duration_s=time.monotonic()-soak_start,
        total_with_checkpoints_s=time.monotonic()-started, counters=counters,
        initial=initial, final=memory(), tail_rss_min_kib=min(ys) if ys else None,
        tail_rss_max_kib=max(ys) if ys else None, tail_rss_slope_kib_per_s=slope,
        max_fds=max(s['fds'] for s in samples), min_fds=min(s['fds'] for s in samples),
        assertions='all active maps empty after each cohort; accounting; stale generations',
        scope='CPU admission on/off routing harness, not production RuntimeGuard hot reload',
        caveat='RSS plateau cannot prove no leaks; no NPU or inference timing measured',
        extension_sha256=hashlib.sha256(Path(__import__('_m4_native').__file__).read_bytes()).hexdigest())
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
