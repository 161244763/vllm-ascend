"""Differential tests and complete Qwen3-8B dataset replay; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prediction(det, ids, chunk):
    st = det.start()
    for i in range(0, len(ids), chunk):
        res = det.feed(st, ids[i:i+chunk])
        if res is not None and res.is_ill:
            return dict(fired_at=res.fired_at, reason=res.reason, finish_only=False)
    res = det.finish(st)
    return dict(fired_at=res.fired_at, reason=res.reason, finish_only=res.is_ill)


def features(st):
    def share(r, n, w):
        return (r.at(n)-r.at(n-w))/w if n >= w else np.nan
    return [st.streams[4].share('A3',1024), st.streams[4].share('A5',512),
            st.streams[4].share('N',512),st.streams[8].share('P',1024),
            share(st.repcum,st.nx,2048),share(st.repcum,st.nx,1024),
            share(st.markcum,st.t,1024),share(st.dupcum,st.nh,16),
            (st.sepcum.at(st.t)-st.sepcum.at(max(st.t-256,0)))/max(min(st.t,256),1)]


def tests(ref, cpp):
    from v35_detector.stream import exact_block
    from _m4_native import exact_check
    rng = np.random.default_rng(935)
    cases = [[], [198], [151668], rng.integers(0,151000,65599).tolist()]
    for period in [1,4,8,31,32,63,64,65,499,500,501,1999,2000,2001]:
        pattern = rng.integers(1000,140000,period).tolist()
        for prefix in [0,17,63]:
            cases.append(rng.integers(1000,140000,prefix).tolist()+pattern*7+[151668]+pattern*4)
    # Boundary reset, SEP-only, sentence history wrap and partial final blocks.
    word = np.flatnonzero((ref.T['norm'] >= 0) & ~ref.T['send'])[:1000].tolist()
    end = int(np.flatnonzero(ref.T['send'])[0])
    sep = int(np.flatnonzero(ref.T['norm'] == -1)[0])
    cases += [[sep]*10001, ([*word[:5], end]*600),
              word*5+[151668]+word*5, word*10, word[:63], word[:65]]
    checked_boundaries = 0
    for ci, ids in enumerate(cases):
        expected = prediction(ref,ids,64)
        for chunk in [1,17,64,129,8192]:
            actual = prediction(cpp,ids,chunk)
            assert actual == expected, (ci,chunk,actual,expected)
        a,b = ref.start(),cpp.start()
        for i in range(0,len(ids),64):
            ref.feed(a,ids[i:i+64]); cpp.feed(b,ids[i:i+64])
            assert (a.t,a.nx,a.answer_at,a.nh,a.streak) == (b.t,b.nx,None if b.answer_at<0 else b.answer_at,b.sentences,b.streak), (ci,i)
            np.testing.assert_allclose(features(a),b.features(),rtol=0,atol=0,equal_nan=True)
            checked_boundaries += 1
            if a.done: break
        assert ref.finish(a) == cpp.finish(b)
        np.testing.assert_allclose(features(a),b.features(),rtol=0,atol=0,equal_nan=True)
    exact_cases=0
    for trial in range(500):
        p=int(rng.integers(1,2010)); copies=int(rng.choice([3,5]))
        pattern=rng.integers(0,200,p).tolist()
        ids=np.array(rng.integers(0,200,int(rng.integers(0,65))).tolist()+pattern*copies,dtype=np.int64)
        if trial%2: ids[int(rng.integers(0,len(ids)))]=-1
        nnew=min(len(ids),int(rng.integers(1,65)))
        mn,mx=32,(500 if copies==5 else 2000)
        for trim in [0,max(0,len(ids)-6064)]:
            arr=ids[trim:]
            a=exact_block(arr,trim,len(ids),nnew,mn,mx,copies)
            b=exact_check(arr.tolist(),nnew,mn,mx,copies)
            assert bool(a)==b,(trial,trim)
            exact_cases+=1
    batch_ids=[cases[i%len(cases)][:1200] for i in range(128)]
    states=[cpp.start() for _ in batch_ids]
    for i in range(max(map(len,batch_ids))):
        cpp.feed_batch(states,[x[i:i+1] for x in batch_ids])
    for st,ids in zip(states,batch_ids):
        r=cpp.finish(st)
        a=dict(fired_at=r.fired_at,reason=r.reason)
        b=prediction(ref,ids,64)
        assert a=={k:b[k] for k in a}
    try: cpp.feed(cpp.start(),[-1])
    except IndexError: pass
    else: raise AssertionError('invalid ID accepted')
    from m4_cpp import RepDetectorCPP
    other=RepDetectorCPP('qwen3')
    try: cpp.feed(other.start(),[1])
    except ValueError: pass
    else: raise AssertionError('foreign state accepted')
    return dict(sequence_cases=len(cases),chunk_sizes=[1,17,64,129,8192],
                feature_boundaries=checked_boundaries,exact_cases=exact_cases,batch_states=128)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(exist_ok=False)
    sys.path[:0]=[str(args.reference),str(args.reference/'vendor/m4/src')]
    from v35_detector import RepDetector,CONFIG_M4
    from m4_cpp import RepDetectorCPP
    from evaluate import records,summarize,sha256
    ref,cpp=RepDetector('qwen3',CONFIG_M4),RepDetectorCPP('qwen3')
    start=time.monotonic()
    save(args.out/'tests.json',tests(ref,cpp))
    print('DIFFERENTIAL TESTS PASSED',flush=True)
    manifest=json.loads((args.data/'MANIFEST.json').read_text())
    hashes={}
    for name in ['qwen3/token_ids.npy','qwen3/records.jsonl','labels_dev.jsonl']:
        hashes[name]=sha256(args.data/name)
        assert hashes[name]==manifest['files'][name]['sha256'],name
    prior={tuple(r['key']):r for r in map(json.loads,args.archive.open()) if r['profile']=='m4'}
    rows=[];fresh=0;total_tokens=0;fired_tokens=0
    with (args.out/'predictions.jsonl').open('x') as output:
        for meta,raw in records(args.data,['qwen3']):
            if meta['model']!='Qwen3-8B':continue
            ids=raw.tolist();key=tuple(meta['key']);p=prior[key]
            actual=prediction(cpp,ids,64)
            expected={k:p.get(k,False) for k in ['fired_at','reason','finish_only']}
            if actual!=expected:
                save(args.out/'MISMATCH.json',dict(key=key,actual=actual,expected=expected))
                raise AssertionError(('archive mismatch',key))
            # Fresh independent Python replay for every record, no historical-only claim.
            current=prediction(ref,ids,64)
            assert current==actual,('fresh mismatch',key,current,actual)
            fresh+=1;total_tokens+=len(ids)
            fired_tokens+=actual['fired_at'] if actual['fired_at']>=0 else len(ids)
            row={**p,**actual,'profile':'m4_cpp'}
            output.write(json.dumps(row)+'\n');rows.append(row)
            if fresh%100==0:
                output.flush()
                save(args.out/'status.json',dict(state='running',records=fresh,elapsed_s=time.monotonic()-start))
                print(f'{fresh} records exact match; {time.monotonic()-start:.1f}s',flush=True)
    assert fresh==6438 and len(prior)==fresh,(fresh,len(prior))
    save(args.out/'quality.json',summarize(rows))
    save(args.out/'status.json',dict(state='complete',records=fresh,fresh_python_matches=fresh,
         archived_matches=fresh,total_tokens=total_tokens,scored_tokens=fired_tokens,
         elapsed_s=time.monotonic()-start,verified_dataset_hashes=hashes,
         archive_sha256=sha256(args.archive),manifest_sha256=sha256(args.data/'MANIFEST.json')))
    print('ALL VALIDATION PASSED',flush=True)


if __name__=='__main__':main()
