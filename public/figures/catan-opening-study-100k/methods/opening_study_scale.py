#!/usr/bin/env python3
"""Resumable compact frozen-policy opening corpus; CPU only, no search.

Default: 100,000 fresh boards, seeds 4,100,000 through 4,199,999.
Every board follows one greedy A1/B1/B2/A2 trajectory, including greedy roads.
Only a completed shard is published. Re-run the identical command to resume.

Example pilot (separate output):
  .venv/bin/python opening_study_scale.py --boards 100 --batch-size 32 \
      --shard-size 50 --out /private/tmp/catan-opening-scale-pilot
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import catan_rl
import numpy as np
import torch

from opening_engine import OpeningEngine, LABELS, SEATS, MODEL_PATH, MODEL_SHA
from opening_network_study import board_features, build_board
from feature_dictionary import geometry_data
from train import split_obs

ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = ROOT/'run/analysis/opening-study-100k-20261029'
SCHEMA = 'opening-study-scale-v1'
SPLIT_SEED = 291037
NEAR_TIE_MARGIN = .001
LOGIT_VERIFY_ATOL = 5e-5
COMPRESSION_LEVEL = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    """Streaming NumPy ZIP members; low compression level favors throughput."""
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=COMPRESSION_LEVEL, allowZip64=True) as archive:
            for name, value in arrays.items():
                array=np.asarray(value)
                assert not array.dtype.hasobject
                with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def split_flags(boards):
    train=np.zeros(boards,dtype=bool)
    train[np.random.default_rng(SPLIT_SEED).permutation(boards)[:4*boards//5]]=True
    return train


def source_provenance():
    paths=['opening_study_scale.py','opening_network_study.py','opening_engine.py','feature_dictionary.py',
           'analyze_number_diversification.py','train.py','gnn.py','adjacency.json','viz/inference.py','viz/geometry.json',
           'catan/src/py.rs','catan/src/features.rs','catan/src/board.rs','catan/src/rules.rs',
           'catan/src/geometry.rs','catan/src/rng.rs','catan/src/state.rs']
    sources={name:digest(ROOT/name) for name in paths}
    module_dir=Path(catan_rl.__file__).resolve().parent
    compiled={str(path):digest(path) for path in sorted(module_dir.glob('*.so'))}
    return dict(sources=sources,compiled_extensions=compiled,torch_version=str(torch.__version__),numpy_version=str(np.__version__))


class BatchedGenerator:
    def __init__(self, engine=None, geometry=None, distance=None):
        self.engine=engine or OpeningEngine()
        self.engine.model.requires_grad_(False)
        if geometry is None or distance is None:
            geometry,distance=geometry_data()
        self.geometry=geometry;self.distance=distance
        self.forward_batches=0;self.single_tie_fallbacks=0

    @torch.inference_mode()
    def score(self,games,kind):
        observations=np.concatenate([game.opening_observation() for game in games],axis=0)
        logits,_=self.engine.model(*split_obs(torch.from_numpy(observations)))
        self.forward_batches+=1
        results=[]
        offset=72 if kind=='settlement' else 0
        for i,game in enumerate(games):
            legal=np.asarray(game.legal(),dtype=np.int64)
            assert len(legal)>0
            assert np.all((legal>=72)&(legal<126)) if offset else np.all(legal<72)
            scores=logits[i,legal].numpy().copy()
            assert np.isfinite(scores).all()
            order=np.lexsort((legal,-scores))
            # Batched matrix multiplication may round differently from a
            # single-row pass. Resolve close ties using the original path and
            # retain its complete score row, preserving greedy consistency.
            if len(order)>1 and float(scores[order[0]]-scores[order[1]])<NEAR_TIE_MARGIN:
                single,_=self.engine.model(*split_obs(torch.from_numpy(observations[i:i+1])))
                self.single_tie_fallbacks+=1
                scores=single[0,legal].numpy().copy()
                order=np.lexsort((legal,-scores))
            results.append(dict(action=int(legal[order[0]]),vertex=(legal-offset).astype(np.uint8),logit=scores.astype(np.float32,copy=False)))
        return results

    @torch.inference_mode()
    def generate(self,seeds):
        """Return compact per-board records for one bounded batch."""
        seeds=[int(seed) for seed in seeds]
        games=[catan_rl.Game(seed=seed) for seed in seeds]
        records=[dict(seed=seed,board=[list(a) for a in game.board_arrays()],history=[],picks=[]) for seed,game in zip(seeds,games)]
        names=descriptions=None
        for stage,label in enumerate(LABELS):
            assert all(game.to_act()==SEATS[stage] for game in games)
            settlement=self.score(games,'settlement')
            for game,record,decision in zip(games,records,settlement):
                history=[list(step) for step in game.history()]
                current_names,features,current_descriptions=board_features(record['board'],history,SEATS[stage],self.geometry,self.distance)
                if names is None:
                    names,descriptions=current_names,current_descriptions
                else:
                    assert names==current_names
                chosen=decision['action']-72
                record['picks'].append(dict(label=label,chosen=chosen,vertex=decision['vertex'],logit=decision['logit'],
                    features=features[decision['vertex']].astype(np.float32)))
                game.apply(decision['action'])
            roads=self.score(games,'road')
            for game,record,decision in zip(games,records,roads):
                road=decision['action']; record['picks'][-1]['road']=road
                game.apply(road)
        for game,record in zip(games,records):
            record['history']=[int(a) for _,a in game.history()]
            assert len(record['history'])==8
        return records,names,descriptions


def pack_shard(records,train):
    """Board-major, then stage-major, with ragged legal candidate rows."""
    offsets=[0];vertices=[];logits=[];features=[]
    for board in records:
        for pick in board['picks']:
            n=len(pick['vertex'])
            assert pick['features'].shape==(n,50) and pick['logit'].shape==(n,)
            offsets.append(offsets[-1]+n)
            vertices.append(pick['vertex']);logits.append(pick['logit']);features.append(pick['features'])
    return dict(board_seed=np.asarray([b['seed'] for b in records],dtype=np.uint64),train=np.asarray(train,dtype=bool),
        board_resources=np.asarray([b['board'][0] for b in records],dtype=np.uint8),
        board_numbers=np.asarray([b['board'][1] for b in records],dtype=np.uint8),
        board_ports=np.asarray([b['board'][2] for b in records],dtype=np.uint8),
        history=np.asarray([b['history'] for b in records],dtype=np.uint8),
        chosen_vertices=np.asarray([[p['chosen'] for p in b['picks']] for b in records],dtype=np.uint8),
        chosen_roads=np.asarray([[p['road'] for p in b['picks']] for b in records],dtype=np.uint8),
        position_ptr=np.asarray(offsets,dtype=np.int64),candidate_vertex=np.concatenate(vertices),
        candidate_logit=np.concatenate(logits),candidate_features=np.concatenate(features))


def validate_shard(arrays,seed_start,expected_train):
    """Cheap complete structural/greedy checks before publication."""
    n=len(expected_train);ptr=arrays['position_ptr'];count=len(arrays['candidate_vertex'])
    np.testing.assert_array_equal(arrays['board_seed'],np.arange(seed_start,seed_start+n,dtype=np.uint64))
    np.testing.assert_array_equal(arrays['train'],expected_train)
    assert ptr.shape==(4*n+1,) and ptr[0]==0 and ptr[-1]==count
    assert np.all(np.diff(ptr)>0)
    assert arrays['candidate_features'].shape==(count,50)
    assert np.isfinite(arrays['candidate_features']).all() and np.isfinite(arrays['candidate_logit']).all()
    np.testing.assert_array_equal(arrays['history'][:,::2],arrays['chosen_vertices']+72)
    np.testing.assert_array_equal(arrays['history'][:,1::2],arrays['chosen_roads'])
    for position,(a,b) in enumerate(zip(ptr[:-1],ptr[1:])):
        vertex=arrays['candidate_vertex'][a:b];score=arrays['candidate_logit'][a:b]
        assert len(np.unique(vertex))==len(vertex)
        choice=int(vertex[np.lexsort((vertex,-score))[0]])
        assert choice==int(arrays['chosen_vertices'][position//4,position%4])
    np.testing.assert_array_equal(arrays['candidate_features'][:,0],arrays['candidate_features'][:,1:6].sum(1))


def verify_against_original(generator,seeds,batch_size):
    """No search: compare batched data to the original independent builder."""
    max_logit_error=0.;max_feature_error=0.;positions=0
    checked=[];all_records=[]
    for start in range(0,len(seeds),batch_size):
        records,_,_=generator.generate(seeds[start:start+batch_size]);all_records.extend(records)
    for batched in all_records:
        original,names,_=build_board(generator.engine,batched['seed'],'verification',generator.geometry,generator.distance)
        assert batched['board']==original['board']
        assert batched['history']==[a for _,a in original['history']]
        for stage,pick in enumerate(batched['picks']):
            reference=original['picks'][stage]
            assert pick['chosen']==reference['chosen'] and pick['road']==reference['road']
            rows={r['vertex']:r for r in reference['candidates']}
            assert set(int(v) for v in pick['vertex'])==set(rows)
            expected_logits=np.asarray([rows[int(v)]['logit'] for v in pick['vertex']],dtype=np.float32)
            expected_features=np.asarray([rows[int(v)]['features'] for v in pick['vertex']],dtype=np.float32)
            error=float(np.max(np.abs(pick['logit']-expected_logits)))
            max_logit_error=max(max_logit_error,error)
            max_feature_error=max(max_feature_error,float(np.max(np.abs(pick['features']-expected_features))))
            np.testing.assert_allclose(pick['logit'],expected_logits,atol=LOGIT_VERIFY_ATOL,rtol=0)
            np.testing.assert_array_equal(pick['features'],expected_features)
            positions+=1
        checked.append(batched['seed'])
    arrays=pack_shard(all_records,np.ones(len(all_records),dtype=bool))
    # The verification seeds need not be contiguous, so perform the remaining
    # pack checks using a temporary contiguous ID vector only.
    packed_for_check=dict(arrays);packed_for_check['board_seed']=np.arange(len(all_records),dtype=np.uint64)
    validate_shard(packed_for_check,0,np.ones(len(all_records),dtype=bool))
    return dict(status='passed',seeds=checked,boards=len(checked),settlement_decisions=positions,
                all_settlement_and_road_choices_match=True,all_legal_sets_match=True,float32_features_match_exactly=True,
                maximum_absolute_logit_difference=max_logit_error,maximum_absolute_feature_difference=max_feature_error,
                logit_absolute_tolerance=LOGIT_VERIFY_ATOL,near_tie_single_forward_threshold=NEAR_TIE_MARGIN)


def read_shard(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def check_seeds(seed,boards):
    if seed<0 or boards<5 or seed+boards>2**53:
        raise ValueError('Need at least five boards and nonnegative exact-integer seeds below 2**53.')
    intervals=[(3100000,3101000,'original 1,000-board study'),(2040000,2040256,'reserved confirmation boards')]
    for a,b,label in intervals:
        if max(a,seed)<min(b,seed+boards):
            raise ValueError('Seed range overlaps '+label)
    gallery=json.loads((ROOT/'run/analysis/opening-gallery-20261004/manifest.json').read_text())
    if any(seed<=row['seed']<seed+boards for row in gallery['boards']):
        raise ValueError('Seed range overlaps the existing gallery.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boards',type=int,default=100000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--shard-size',type=int,default=1000)
    parser.add_argument('--out',type=Path,default=DEFAULT_DEST)
    parser.add_argument('--seed',type=int,default=4100000)
    parser.add_argument('--verify-boards',type=int,default=8,help='Startup comparison against the independent single-board implementation (minimum 4).')
    args=parser.parse_args()
    if args.batch_size<1 or args.shard_size<1 or args.verify_boards<4:
        parser.error('batch/shard sizes must be positive; --verify-boards must be at least 4')
    check_seeds(args.seed,args.boards)
    dest=args.out.resolve()
    original=(ROOT/'run/analysis/opening-study-20261028').resolve()
    if dest==original or original in dest.parents:
        parser.error('Use a new output directory, outside the original 1,000-board study.')
    provenance=source_provenance()
    config=dict(boards=args.boards,seed_start=args.seed,seed_end=args.seed+args.boards-1,batch_size=args.batch_size,
        shard_size=args.shard_size,split_rng_seed=SPLIT_SEED,train_boards=4*args.boards//5,test_boards=args.boards-4*args.boards//5,
        compression_level=COMPRESSION_LEVEL,near_tie_margin=NEAR_TIE_MARGIN,verify_boards=args.verify_boards)
    manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['schema']!=SCHEMA or manifest['config']!=config or manifest['provenance']!=provenance:
            raise RuntimeError('Resume configuration or source provenance differs. Use the original command/source or a fresh output directory.')
    else:
        dest.mkdir(parents=True,exist_ok=True);(dest/'shards').mkdir(exist_ok=True)
        for name in provenance['sources']:
            target=dest/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        manifest=dict(schema=SCHEMA,status='initializing',created_utc=utc_now(),updated_utc=utc_now(),config=config,
            provenance=provenance,labels=list(LABELS),seats=list(SEATS),model=dict(name='Gen37',sha256=MODEL_SHA,path=str(MODEL_PATH),mode='greedy_network',search_simulations=0),
            completed_boards=0,completed_settlement_decisions=0,candidate_count=0,stored_bytes=0,shards=[],
            feature_names=[],feature_descriptions={},geometry=None,
            precision=dict(candidate_features='float32, rounded from existing float64 descriptive calculations',candidate_logit='float32 neural output; batching can change final floating-point bits',
                decision_guard='Rows whose best-two batched legal logits are less than 0.001 apart are recomputed individually, and the entire individual logit row is retained.'),
            array_schema=dict(board_seed='[N] uint64, absolute board generator seeds',train='[N] bool, immutable whole-board train split',
                board_resources='[N,19] uint8; wood0 brick1 sheep2 wheat3 ore4 desert5',board_numbers='[N,19] uint8',board_ports='[N,9] uint8; resources0..4, generic5; geometry port order',
                history='[N,8] uint8 setup action IDs; settlement72+vertex and road0..71 alternating A1/B1/B2/A2; all steps have referee kind1',
                chosen_vertices='[N,4] uint8',chosen_roads='[N,4] uint8',position_ptr='[4*N+1] int64; board i, stage s uses candidate rows ptr[4*i+s]:ptr[4*i+s+1]',
                candidate_vertex='[C] uint8, legal vertex ID',candidate_logit='[C] float32, raw legal settlement score',candidate_features='[C,50] float32, raw units indexed by feature_names'),
            interpretation='One deterministic greedy opening trajectory per generated board. Scores describe the frozen raw policy, not searched play, causal resource importance or winning probabilities.',
            resume='Completed shards and their hashes are immutable. A shard left after a crash before manifest publication is validated and adopted on resume.',
            regenerate='Board arrays plus eight actions reproduce all observations/codes; detailed neuron traces can be regenerated only for selected examples.')
        atomic_json(manifest_path,manifest)
    # Validate hashes of previously completed artifacts without loading them.
    for shard in manifest['shards']:
        path=dest/shard['path']
        if not path.exists() or digest(path)!=shard['sha256']:
            raise RuntimeError('Saved shard is missing or changed: '+str(path))
    if manifest['status']=='complete':
        print(f'Already complete: {manifest["completed_boards"]} boards at {dest}',flush=True);return
    generator=BatchedGenerator();train=split_flags(args.boards)
    if 'verification' not in manifest:
        # A fixed spread of requested-range boards is independently compared
        # before corpus generation, including the two range endpoints.
        count=min(args.verify_boards,args.boards)
        check_indices=np.linspace(0,args.boards-1,count,dtype=int)
        verification=verify_against_original(generator,[args.seed+int(i) for i in check_indices],args.batch_size)
        manifest['verification']=verification
        print('Batched/single verification passed: '+json.dumps(verification),flush=True)
    generator.forward_batches=0;generator.single_tie_fallbacks=0
    manifest.update(status='running',updated_utc=utc_now())
    atomic_json(manifest_path,manifest)
    start_time=time.monotonic();previous_boards=manifest['completed_boards']
    for shard_index,begin in enumerate(range(0,args.boards,args.shard_size)):
        end=min(args.boards,begin+args.shard_size)
        if shard_index<len(manifest['shards']):
            continue
        path=dest/'shards'/f'shard-{shard_index:05d}.npz'
        shard_started=time.monotonic();from_orphan=path.exists();records=[]
        calls_before=generator.forward_batches;ties_before=generator.single_tie_fallbacks
        if from_orphan:
            packed=read_shard(path)
            first=catan_rl.Game(seed=args.seed+begin)
            names,_,descriptions=board_features([list(a) for a in first.board_arrays()],[],0,generator.geometry,generator.distance)
        else:
            last_progress=time.monotonic()
            for batch_start in range(begin,end,args.batch_size):
                batch_end=min(end,batch_start+args.batch_size)
                batch,names,descriptions=generator.generate(range(args.seed+batch_start,args.seed+batch_end))
                records.extend(batch)
                if time.monotonic()-last_progress>=10:
                    print(f'shard {shard_index}: {batch_end-begin}/{end-begin} boards generated',flush=True)
                    last_progress=time.monotonic()
            packed=pack_shard(records,train[begin:end]);del records
        validate_shard(packed,args.seed+begin,train[begin:end])
        if manifest['feature_names']:
            assert manifest['feature_names']==names
        else:
            manifest.update(feature_names=names,feature_descriptions=descriptions,geometry=generator.geometry)
        generate_seconds=time.monotonic()-shard_started
        write_started=time.monotonic()
        if not from_orphan:
            atomic_npz(path,packed)
        row=dict(index=shard_index,path=str(path.relative_to(dest)),board_count=end-begin,seed_start=args.seed+begin,seed_end=args.seed+end-1,
            train_boards=int(train[begin:end].sum()),test_boards=int((~train[begin:end]).sum()),candidate_count=len(packed['candidate_vertex']),
            bytes=path.stat().st_size,sha256=digest(path),generation_seconds=round(generate_seconds,6),write_seconds=round(time.monotonic()-write_started,6),
            forward_batches=generator.forward_batches-calls_before,single_tie_fallbacks=generator.single_tie_fallbacks-ties_before,adopted_after_interruption=from_orphan)
        del packed
        manifest['shards'].append(row)
        manifest.update(completed_boards=end,completed_settlement_decisions=4*end,candidate_count=sum(s['candidate_count'] for s in manifest['shards']),
            stored_bytes=sum(s['bytes'] for s in manifest['shards']),updated_utc=utc_now(),elapsed_generation_and_write_seconds=sum(s['generation_seconds']+s['write_seconds'] for s in manifest['shards']))
        if end==args.boards:
            assert digest(MODEL_PATH)==MODEL_SHA
            assert source_provenance()==provenance
            manifest['status']='complete'
        atomic_json(manifest_path,manifest)
        elapsed=time.monotonic()-start_time;done=end-previous_boards
        rate=done/max(elapsed,.001)
        eta=(args.boards-end)/rate
        print(f'{end}/{args.boards} boards; {manifest["candidate_count"]:,} legal candidates; {manifest["stored_bytes"]/1e6:.1f} MB; {rate:.1f} boards/s; ETA {eta/60:.1f} min',flush=True)
    print(dest,flush=True)


if __name__=='__main__':
    main()
