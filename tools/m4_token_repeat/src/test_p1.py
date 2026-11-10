"""Differential tests for P1 native events and its actual request adapter.

Valid-input semantics are checked against fresh frozen Python and the original
C++ feed path. Robustness changes are explicitly labelled, never used as a
performance claim. This suite needs no model inference or NPU.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from m4_cpp import RepDetectorCPP
from perf_session import CPUBackend, LegacyBackend, PerfSession
from v35_detector import CONFIG_M4, RepDetector
from validate import features


def snapshot(st):
    return dict(t=st.t, nx=st.nx, answer_at=st.answer_at, streak=st.streak,
                sentences=st.sentences, done=st.done,
                fired_at=st.fired_at, reason=st.reason, features=st.features())


def compare_state(ref, cpp):
    assert (ref.t, ref.nx, ref.answer_at, ref.nh, ref.streak, ref.done) == (
        cpp.t, cpp.nx, None if cpp.answer_at < 0 else cpp.answer_at,
        cpp.sentences, cpp.streak, cpp.done), (
            'state mismatch', ref.t, ref.nx, ref.answer_at, ref.nh, ref.streak, ref.done,
            cpp.t, cpp.nx, cpp.answer_at, cpp.sentences, cpp.streak, cpp.done)
    np.testing.assert_allclose(features(ref), cpp.features(), rtol=0, atol=0, equal_nan=True)


def serial(det, ids, chunk):
    st = det.start()
    result = None
    for pos in range(0, len(ids), chunk):
        result = det.feed(st, ids[pos:pos + chunk])
        if result is not None and result.is_ill:
            return st, dict(fired_at=result.fired_at, reason=result.reason, finish_only=False)
    result = det.finish(st)
    return st, dict(fired_at=result.fired_at, reason=result.reason, finish_only=result.is_ill)


def sparse(det, ids, chunk):
    st = det.start()
    for pos in range(0, len(ids), chunk):
        events = det.feed_events([st], [ids[pos:pos + chunk]])
        if events:
            assert len(events) == 1 and events[0][0] == 0
            return st, dict(fired_at=events[0][1], reason=events[0][2], finish_only=False)
    result = det.finish(st)
    return st, dict(fired_at=result.fired_at, reason=result.reason, finish_only=result.is_ill)


def run_equivalence():
    ref, cpp = RepDetector('qwen3', CONFIG_M4), RepDetectorCPP('qwen3')
    rng = np.random.default_rng(935)
    lengths = [0, 1, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097]
    cases = []
    for n in lengths:
        cases.append(rng.integers(1000, 140000, n).tolist())
        cases.append([64] * n)
    for n in [63, 64, 65, 127, 128, 129, 1023, 4096]:
        cases.append(rng.integers(1000, 140000, n - 1).tolist() + [151668])
        cases.append(rng.integers(1000, 140000, n // 2).tolist() + [151668] +
                     rng.integers(1000, 140000, n - n // 2 - 1).tolist())
    words = np.flatnonzero((ref.T['norm'] >= 0) & ~ref.T['send'])[:1000].tolist()
    end = int(np.flatnonzero(ref.T['send'])[0])
    sep = int(np.flatnonzero(ref.T['norm'] == -1)[0])
    cases += [[sep] * 10001, [*sum(([w, end] for w in words[:5]), [])] * 600,
              words * 5 + [151668] + words * 5, words * 10]
    chunks = [1, 17, 63, 64, 65, 129, 8192]
    chunk_comparisons = boundary_comparisons = 0
    for index, ids in enumerate(cases):
        _, expected = serial(ref, ids, 64)
        for chunk in chunks:
            a, old = serial(cpp, ids, chunk)
            b, actual = sparse(cpp, ids, chunk)
            assert actual == expected == old, (index, chunk, actual, expected, old)
            # Compare every complete 64-token prefix, not only the final state.
            for end in range(64, len(ids) + 1, 64):
                reference_state, _ = serial(ref, ids[:end], end)
                sparse_state, _ = sparse(cpp, ids[:end], end)
                compare_state(reference_state, sparse_state)
                boundary_comparisons += 1
            compare_state(serial(ref, ids, chunk)[0], b)
            # Every old and new request must finish with exactly the same state.
            assert a.t == b.t and a.reason == b.reason and a.fired_at == b.fired_at
            chunk_comparisons += 1
    return dict(cases=len(cases), chunk_comparisons=chunk_comparisons,
                boundary_comparisons=boundary_comparisons,
                lengths=lengths, chunks=chunks, think_end_boundary_tests=True)


def run_batch_and_errors():
    cpp = RepDetectorCPP('qwen3')
    rng = np.random.default_rng(411)
    for concurrency in [1, 16, 64, 128]:
        inputs = []
        for i in range(concurrency):
            n = int(rng.choice([0, 1, 63, 64, 65, 127, 128, 129, 4097]))
            inputs.append(([64] * n) if i % 4 == 0 else rng.integers(1000, 140000, n).tolist())
        expected = [serial(cpp, ids, 17)[1] for ids in inputs]
        states = [cpp.start() for _ in inputs]
        pos = [0] * concurrency
        actual = [None] * concurrency
        order = list(range(concurrency))
        rng.shuffle(order)
        step = 0
        while any(x is None for x in actual):
            if step % 3 == 0:
                order = order[1:] + order[:1]
            slots, ids = [], []
            for slot in order:
                if actual[slot] is not None:
                    continue
                size = int(rng.choice([0, 1, 17, 63, 64, 65, 129]))
                end = min(pos[slot] + size, len(inputs[slot]))
                slots.append(slot)
                ids.append(inputs[slot][pos[slot]:end])
                pos[slot] = end
            events = cpp.feed_events([states[s] for s in slots], ids)
            previous = -1
            for local, fired_at, reason in events:
                slot = slots[local]
                assert actual[slot] is None and local > previous
                previous = local
                actual[slot] = dict(fired_at=fired_at, reason=reason, finish_only=False)
            for local, slot in enumerate(slots):
                if actual[slot] is None and pos[slot] == len(inputs[slot]):
                    result = cpp.finish(states[slot])
                    actual[slot] = dict(fired_at=result.fired_at, reason=result.reason,
                                        finish_only=result.is_ill)
            step += 1
            assert step < 2000
        assert actual == expected, concurrency
        for st, result in zip(states, actual):
            # Legacy feed_batch continues to materialize all saved first alerts.
            old = cpp.feed_batch([st], [[]])[0]
            if result['fired_at'] >= 0:
                assert old is not None and old.is_ill
                assert (old.fired_at, old.reason) == (result['fired_at'], result['reason'])
            else:
                assert old is None
            if result['fired_at'] >= 0 and not result['finish_only']:
                # The sparse event already saved the first alert. feed() does not
                # consume an unread suffix after done and returns that saved alert.
                before = st.t
                saved = cpp.feed(st, [64])
                assert saved is not None and saved.fired_at == result['fired_at']
                assert st.t == before
                assert cpp.monitor(st).fired_at == result['fired_at']
    def compare_continuation(good, reference):
        # Both sides have the same lifecycle: feed the same prefix and do not
        # finish. serial() would finish and set Python done for a no-alert stream.
        reference_state = reference.start()
        reference.feed(reference_state, [64] * 64)
        pristine = cpp.start()
        assert cpp.feed_events([good, pristine], [[64] * 64, []]) == []
        compare_state(reference_state, good)

    # Invalid states must not crash or partially advance another valid request.
    # Every scenario owns fresh native/Python states, including the bad state.
    for bad_kind in ['none', 'foreign']:
        good = cpp.start()
        other = RepDetectorCPP('qwen3').start() if bad_kind == 'foreign' else None
        try:
            cpp.feed_events([good, other], [[64], [64]])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid state accepted')
        assert good.t == 0
        # Pending word/sentence state must also remain untouched. A t==0 check
        # alone misses mutations before the first feature boundary.
        compare_continuation(good, RepDetector('qwen3', CONFIG_M4))
    foreign = RepDetectorCPP('qwen3').start()
    for states in [[None], [foreign]]:
        try:
            cpp.feed_batch(states, [[]])
        except ValueError:
            pass
        else:
            raise AssertionError('legacy batch invalid state accepted')
    good = cpp.start()
    for bad in [[good], [good, good]]:
        try:
            cpp.feed_events(bad, [] if len(bad) == 1 else [[64], [64]])
        except ValueError:
            pass
        else:
            raise AssertionError('bad length or duplicate accepted')
        assert good.t == 0
    for token in [-1, len(cpp.T['norm'])]:
        good = cpp.start()
        try:
            cpp.feed_events([good], [[token]])
        except IndexError:
            pass
        else:
            raise AssertionError('invalid token accepted')
        assert good.t == 0
        compare_continuation(good, RepDetector('qwen3', CONFIG_M4))
    # The unread suffix after an alert is not converted or validated.
    early = cpp.start()
    events = cpp.feed_events([early], [[64] * 128 + [-1]])
    assert events[0][1] == 128 and early.t == 128
    assert cpp.feed_events([early], [[-1]]) == []
    # Empty native submissions and empty per-request chunks are real supported inputs.
    assert cpp.feed_events([], []) == []
    empty = cpp.start()
    assert cpp.feed_events([empty], [[]]) == []
    return dict(concurrency=[1, 16, 64, 128], shuffled_batch=True,
                empty_steps=True, invalid_states_and_tokens=True,
                alert_suffix_lazy=True, duplicate_state_rejected=True)


def run_lifecycle():
    detector = RepDetectorCPP('qwen3')
    common = dict(retire_rows=True, retain_retired_rows=True)
    legacy = PerfSession({}, factory=lambda _: LegacyBackend(detector), **common)
    candidate = PerfSession({}, factory=CPUBackend, **common)
    # Empty sampled is a no-op, not implicit zero-token request registration.
    legacy.sampled(['b'], [[]])
    candidate.sampled(['b'], [[]])
    assert not legacy.rows and not legacy.states and not legacy.retired_rows
    assert not candidate.rows and not candidate.states and not candidate.retired_rows
    for session in [legacy, candidate]:
        session.register(['empty'])
        session.finish(['empty'])
        session.finished(['empty'])
        assert session.retired_rows[-1][2]['generated'] == 0
        assert session.retired_rows[-1][2]['received'] == 0
    workload = [('a', [64] * 63), ('a', [64]), ('a', [64] * 64),
                ('b', [1, 2, 3]), ('a', [64] * 50), ('a', [])]
    for key, ids in workload:
        legacy.sampled([key], [ids])
        candidate.sampled([key], [ids])
        if ids:
            assert legacy.inspect([key]) == candidate.inspect([key])
        if key == 'b':
            legacy.finished([key])
            candidate.finished([key])
            assert legacy.retired_rows[-1][2] == candidate.retired_rows[-1][2]
    assert legacy.inspect(['a'])['a']['alert_at'] == 128
    assert candidate.inspect(['a'])['a']['generated'] == 178
    assert candidate.inspect(['a'])['a']['scored'] == 128
    assert legacy.retired_rows[0][2] == candidate.retired_rows[0][2]
    for session in [legacy, candidate]:
        session.finished(['a', 'a'])
        assert session.snapshot()['live_states'] == 0
        session.register(['a'])
        session.sampled(['a'], [[64] * 128])
        assert session.rows['a'] == dict(generated=128, scored=128, alert_at=128)
        assert session.snapshot()['retired_requests'] == 3
        session.finish(['a'])
        session.finished(['a'])
        assert session.snapshot()['retired_requests'] == 4
    # Explicit incarnation identity makes late finish/cancel harmless after reuse.
    safe = PerfSession({}, factory=CPUBackend)
    first = safe.register(['x'])[0]
    safe.sampled(['x'], [[64] * 127], [first])
    safe.finish(['x'], [first])
    safe.finished(['x'], [first])
    second = safe.register(['x'])[0]
    safe.sampled(['x'], [[64] * 65], [second])
    before = safe.inspect(['x'])
    safe.finish(['x'], [first])
    safe.finished(['x'], [first])
    assert safe.inspect(['x']) == before
    assert safe.received['x'] == 65
    safe.finish(['x'], [second])
    safe.finished(['x'], [second])
    # Duplicate keys are rejected before generated counters or state creation.
    for duplicate in [(['a', 'a'], [[64], [64]]), (['new', 'new'], [[], [64]])]:
        guarded = PerfSession({}, factory=CPUBackend)
        guarded.sampled(['a'], [[64]])
        before = (dict(guarded.rows), dict(guarded.received), dict(guarded.states))
        try:
            guarded.sampled(*duplicate)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate request key accepted')
        assert before == (guarded.rows, guarded.received, guarded.states)
    # finish-only equality with the actual adapter; an empty finish is not an alert.
    for ids in [[], [64] * 127, [64] * 511, [64] * 512]:
        ref = RepDetector('qwen3', CONFIG_M4)
        expected = serial(ref, ids, 1)[1]
        a2 = PerfSession({}, factory=CPUBackend)
        a2.register(['x'])
        a2.sampled(['x'], [ids])
        a2.finish(['x'])
        actual = a2.inspect(['x'])['x']
        assert (actual['fired_at'], actual['reason'], actual['finish_only']) == (
            expected['fired_at'], expected['reason'], expected['finish_only'])
        assert (actual['alert_at'] is not None) == (expected['fired_at'] >= 0)
        if actual['alert_at'] is not None:
            assert actual['alert_at'] == expected['fired_at']
        accepted = expected['fired_at'] if expected['fired_at'] >= 0 else len(ids)
        assert actual['received'] == accepted
        assert actual['scored'] == accepted
        assert actual['computed_tokens'] == accepted
        assert actual['generated'] == len(ids)
        a2.finish(['x'])
        a2.finished(['x'])
    return dict(cancel=True, repeated_cleanup=True, id_reuse=True,
                explicit_generation_late_callbacks=True, duplicate_keys_no_side_effect=True,
                empty_explicit_registration=True, finish_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, fn in [('equivalence', run_equivalence),
                     ('batch_and_errors', run_batch_and_errors),
                     ('lifecycle', run_lifecycle)]:
        results[name] = fn()
        print(name, json.dumps(results[name]), flush=True)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('P1 DIFFERENTIAL TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
