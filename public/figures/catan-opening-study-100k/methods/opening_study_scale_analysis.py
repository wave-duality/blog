#!/usr/bin/env python3
"""Bounded-memory opening-policy analysis; never trains the policy or runs search.

Two passes over compact shards: training sufficient statistics, then held-out
evaluation and board-cluster uncertainty. All model choices are fixed in source.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

from opening_network_study import RESOURCE, feature_models

ALPHA = .001
LABELS = ['A1', 'B1', 'B2', 'A2']
REFERENCE_SUPPLY = np.array([4, 3, 4, 4, 3]) * 58 / 18
PIPS = np.array([0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1])
Z95 = NormalDist().inv_cdf(.975)
Z40 = NormalDist().inv_cdf(1 - .05 / (2 * 40))


def centered_moments(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(0); y = y - y.mean()
    return x.T @ x / len(x), x.T @ y / len(x), float(y @ y / len(y))


def fit_moments(cov, rhs, columns, alpha=ALPHA):
    cov = np.asarray(cov, dtype=float); rhs = np.asarray(rhs, dtype=float)
    rms = np.sqrt(np.maximum(np.diag(cov), 0))
    scale = np.where(rms > 1e-10, rms, 1.)
    c = cov / scale[:, None] / scale[None, :]
    b = rhs / scale
    keep = []; omitted = []
    for j, name in enumerate(columns):
        trial = keep + [j]
        if rms[j] > 1e-10 and np.linalg.matrix_rank(c[np.ix_(trial, trial)], tol=1e-9) == len(trial):
            keep.append(j)
        else:
            omitted.append(name)
    std = np.zeros(len(columns))
    if keep:
        std[keep] = np.linalg.solve(c[np.ix_(keep, keep)] + alpha * np.eye(len(keep)), b[keep])
    return dict(raw=std / scale, standardized=std, scale=scale,
                active_indices=keep, omitted=omitted)


def enriched_stats(chosen, baseline):
    chosen = np.asarray(chosen, dtype=float); baseline = np.asarray(baseline, dtype=float)
    assert chosen.shape == baseline.shape and chosen.ndim == 2
    n = len(chosen)
    if not n:
        return dict(n=0)
    assert np.all(chosen.sum(1) > 0)
    np.testing.assert_allclose(baseline.sum(1), 1, atol=1e-7)
    shares = chosen / chosen.sum(1)[:, None]
    a = shares.mean(0); b = baseline.mean(0); delta = a - b
    dse = (shares - baseline).std(0, ddof=1) / np.sqrt(n) if n > 1 else None
    ratios = []; intervals = []
    for j in range(chosen.shape[1]):
        if b[j] <= 0:
            ratios.append(None); intervals.append([None, None]); continue
        ratio = a[j] / b[j]; ratios.append(float(ratio))
        if a[j] > 0 and n > 1:
            influence = shares[:, j] / a[j] - baseline[:, j] / b[j]
            se = influence.std(ddof=1) / np.sqrt(n)
            intervals.append((ratio * np.exp(np.array([-1, 1]) * Z95 * se)).tolist())
        else:
            intervals.append([None, None])
    return dict(n=n, chosen_share=a.tolist(), baseline_share=b.tolist(),
                ratio_of_means=ratios, difference=delta.tolist(), ratio_ci95=intervals,
                difference_ci95=np.column_stack([delta-Z95*dse, delta+Z95*dse]).tolist() if n > 1 else [None]*chosen.shape[1])


def score_covariance(score_sum, score_outer, inverse, scale, n):
    """Raw-coefficient covariance from RMS-aware standardized board scores."""
    assert n > 1
    meat = (score_outer - np.outer(score_sum, score_sum)/n) / (n*(n-1))
    transformed = inverse @ meat @ inverse.T
    return transformed / scale[:, None] / scale[None, :]


def model_specs():
    old = feature_models()
    numbers = ['unique_local_numbers', 'unique_combined_numbers', 'pips_overlapping_owned_numbers', 'combined_production_variance']
    specs = {'total_pips': old['total_pips']}
    specs.update({r+'_alone': [r+'_pips'] for r in RESOURCE})
    for k in ['resource_pips', 'resource_coverage', 'resource_coverage_ports']:
        specs[k] = old[k]
    for r in RESOURCE:
        specs['without_'+r] = [c for c in old['resource_pips'] if c != r+'_pips']
    specs['resource_coverage_ports_numbers'] = old['resource_coverage_ports'] + numbers
    specs['full_context'] = old['full_context']
    extra = []
    for suffix in ['owned_production_interaction', 'missing_resource_interaction', 'owned_port_interaction', 'port', 'port_access_interaction', 'starting_cards']:
        extra += [r+'_'+suffix for r in RESOURCE]
    specs['resource_context'] = old['full_context'] + extra
    specs['board_scarcity'] = specs['resource_context'] + [r+'_board_scarcity_interaction' for r in RESOURCE]
    specs['legal_scarcity'] = specs['resource_context'] + [r+'_legal_scarcity_interaction' for r in RESOURCE]
    return specs


def stage_data(shard, stage, names, union_names, geometry):
    n = len(shard['board_seed']); ptr = shard['position_ptr']
    assert shard['train'].dtype == np.bool_, 'Split flags must be Boolean.'
    lo = ptr[stage::4][0:n]; hi = ptr[stage+1::4][0:n]
    count = hi - lo
    assert np.all(count > 0)
    rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
    starts = np.r_[0, np.cumsum(count)[:-1]]
    group = np.repeat(np.arange(n), count)
    f = shard['candidate_features'][rows].astype(np.float64)
    y = shard['candidate_logit'][rows].astype(np.float64)
    vertex = shard['candidate_vertex'][rows]
    byname = {name:f[:, j] for j, name in enumerate(names)}
    rp = np.column_stack([byname[r+'_pips'] for r in RESOURCE])
    own = np.column_stack([byname['combined_'+r+'_pips']-byname[r+'_pips'] for r in RESOURCE])[starts]
    supply = np.column_stack([(PIPS[shard['board_numbers']] * (shard['board_resources']==r)).sum(1) for r in range(5)])
    legal_mean = np.add.reduceat(rp, starts, axis=0) / count[:, None]
    owned_port = np.zeros((n, 5))
    if stage >= 2:
        first = shard['chosen_vertices'][:, 1 if stage == 2 else 0]
        for port, vs in enumerate(geometry['port_vertices']):
            touches = np.isin(first, vs)
            for r in range(5):
                owned_port[:, r] += touches & (shard['board_ports'][:, port] == r)
    for j, r in enumerate(RESOURCE):
        byname[r+'_owned_production_interaction'] = rp[:, j] * own[group, j]
        byname[r+'_missing_resource_interaction'] = rp[:, j] * (own[group, j] == 0)
        byname[r+'_owned_port_interaction'] = rp[:, j] * owned_port[group, j]
        byname[r+'_port_access_interaction'] = rp[:, j] / (1+byname[r+'_port_distance'])
        byname[r+'_board_scarcity_interaction'] = rp[:, j] * (REFERENCE_SUPPLY[j]-supply[group, j])
        byname[r+'_legal_scarcity_interaction'] = rp[:, j] * (REFERENCE_SUPPLY[j]/9-legal_mean[group, j])
    x = np.column_stack([byname[k] for k in union_names])
    mean = np.add.reduceat(x, starts, axis=0) / count[:, None]
    ymean = np.add.reduceat(y, starts) / count
    xc = x - mean[group]; yc = y - ymean[group]
    target = np.add.reduceat(yc*yc, starts) / count
    return dict(x=xc, y=yc, vertex=vertex, count=count, starts=starts, group=group, target=target,
                rp=rp, own=own, owned_port=owned_port, supply=supply, legal_mean=legal_mean,
                adjacent=byname['adjacent_hexes'], total=byname['total_pips'], train=shard['train'],
                chosen=shard['chosen_vertices'][:, stage], seed=shard['board_seed'])


def shard_paths(source, manifest):
    # Shards are always enumerated from the manifest, never from unfinished files.
    return [source / row['path'] for row in manifest['shards']]


def choose_from_scores(score, vertex, starts, group):
    maxima = np.maximum.reduceat(score, starts)
    return np.minimum.reduceat(np.where(score == maxima[group], vertex, 255), starts)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def interval_mean(x):
    x = np.asarray(x, dtype=float); mean = float(x.mean())
    if len(x) < 2:
        raise ValueError('An uncertainty interval needs at least two independent boards.')
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    return mean, [mean-Z95*se, mean+Z95*se]


def r2_stats(error, target):
    e = float(error.mean()); q = float(target.mean())
    if len(error) < 2 or q <= 0:
        raise ValueError('R² intervals need at least two boards and positive target variance.')
    value = 1-e/q
    influence = -(error - (e/q)*target) / q
    se = float(influence.std(ddof=1)/np.sqrt(len(error)))
    return value, [value-Z95*se, value+Z95*se]


def paired_r2_difference(reduced_error, full_error, target):
    d = reduced_error-full_error; q = target.mean()
    if len(d) < 2 or q <= 0:
        raise ValueError('Paired R² intervals need at least two boards and positive target variance.')
    value = float(d.mean()/q)
    influence = (d-value*target)/q
    se = float(influence.std(ddof=1)/np.sqrt(len(d)))
    return dict(delta_r2=value, ci95=[value-Z95*se, value+Z95*se])


def descriptive_records(d):
    starts, count, group = d['starts'], d['count'], d['group']
    chosen_rows = np.flatnonzero(d['vertex'] == d['chosen'][group])
    assert len(chosen_rows) == len(count)
    chosen = d['rp'][chosen_rows]
    assert np.all(chosen.sum(1)>0)
    positive = d['total']>0
    shares = np.divide(d['rp'], d['total'][:, None], out=np.zeros_like(d['rp']), where=positive[:, None])
    positive_count = np.add.reduceat(positive.astype(int), starts)
    baseline = np.add.reduceat(shares, starts, axis=0) / positive_count[:, None]
    matched = positive & (d['total'] == d['total'][chosen_rows][group]) & (d['adjacent'] == d['adjacent'][chosen_rows][group])
    matched_count = np.add.reduceat(matched.astype(int), starts)
    matched_baseline = np.add.reduceat(shares*matched[:, None], starts, axis=0) / matched_count[:, None]
    return dict(seed=d['seed'], train=d['train'], chosen=chosen, baseline=baseline,
                matched_baseline=matched_baseline, matched_count=matched_count,
                own=d['own'], owned_port=d['owned_port'], supply=d['supply'], legal_mean=d['legal_mean'],
                zero_pip_candidates=count-positive_count)


def analyze(source, destination):
    started = time.monotonic(); destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='complete', 'Generation must finish before final analysis.'
    if min(manifest['config']['train_boards'],manifest['config']['test_boards']) < 2:
        raise ValueError('This analysis requires at least two training and two held-out boards.')
    specs = model_specs(); union_names = list(dict.fromkeys(k for cols in specs.values() for k in cols))
    names = manifest['feature_names']; geometry = manifest['geometry']; paths = shard_paths(source, manifest)
    k = len(union_names)
    sums = [dict(cov=np.zeros((k,k)), rhs=np.zeros(k), n=0) for _ in LABELS]
    for si, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            shard = {key:archive[key] for key in archive.files}
        for stage in range(4):
            d = stage_data(shard, stage, names, union_names, geometry)
            w = d['train'][d['group']] / d['count'][d['group']]
            sums[stage]['cov'] += d['x'].T @ (d['x']*w[:, None])
            sums[stage]['rhs'] += d['x'].T @ (d['y']*w)
            sums[stage]['n'] += int(d['train'].sum())
        if si % 10 == 0 or si+1 == len(paths):
            print(f'Training moments: {si+1}/{len(paths)} shards ({time.monotonic()-started:.1f}s)', flush=True)
    fits = []
    for stage, agg in enumerate(sums):
        stage_fits = {}
        for key, cols in specs.items():
            ix = np.array([union_names.index(c) for c in cols])
            cov = agg['cov'][np.ix_(ix,ix)]/agg['n']; rhs=agg['rhs'][ix]/agg['n']
            fit = fit_moments(cov, rhs, cols); active=np.array(fit['active_indices'], dtype=int)
            c = cov[np.ix_(active,active)]/fit['scale'][active,None]/fit['scale'][None,active]
            fit.update(ix=ix, active=active, inverse=np.linalg.inv(c+ALPHA*np.eye(len(active))),
                       score_sum=np.zeros(len(active)), score_outer=np.zeros((len(active),len(active))),
                       errors=[], hits=[], gaps=[])
            stage_fits[key] = fit
        fits.append(stage_fits)
    targets = [[] for _ in LABELS]; descriptive = [[] for _ in LABELS]
    validation = dict(candidate_argmax_matches=0, candidate_choices_checked=0,
                      max_training_score_mean=0., a1_scarcity_prediction_difference=0.)
    for si,path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            shard={key:archive[key] for key in archive.files}
        for stage in range(4):
            d=stage_data(shard,stage,names,union_names,geometry)
            n=d['count']; starts=d['starts']; group=d['group']; tr=d['train']; te=~tr
            actual=choose_from_scores(d['y'], d['vertex'], starts, group)
            np.testing.assert_array_equal(actual,d['chosen'])
            validation['candidate_argmax_matches'] += len(actual); validation['candidate_choices_checked'] += len(actual)
            targets[stage].append(d['target'][te]); descriptive[stage].append(descriptive_records(d))
            scarcity_predictions={}
            for key,fit in fits[stage].items():
                xx=d['x'][:,fit['ix']]; prediction=xx@fit['raw']; residual=d['y']-prediction
                error=np.add.reduceat(residual*residual,starts)/n
                choice=choose_from_scores(prediction,d['vertex'],starts,group)
                maxima=np.maximum.reduceat(d['y'],starts)
                selected=np.add.reduceat(d['y']*(d['vertex']==choice[group]),starts)
                fit['errors'].append(error[te]); fit['hits'].append((choice==d['chosen'])[te]); fit['gaps'].append((maxima-selected)[te])
                a=fit['active']; z=xx[:,a]/fit['scale'][a]
                # The second term includes uncertainty in the fitted RMS scales.
                scores=np.add.reduceat(z*residual[:,None],starts,axis=0)/n[:,None]
                scores-=ALPHA*np.add.reduceat(z*z,starts,axis=0)/n[:,None]*fit['standardized'][a]
                scores=scores[tr]
                fit['score_sum']+=scores.sum(0); fit['score_outer']+=scores.T@scores
                if stage==0 and key in ('board_scarcity','legal_scarcity'):
                    scarcity_predictions[key]=prediction
            if stage==0:
                delta=float(np.max(np.abs(scarcity_predictions['board_scarcity']-scarcity_predictions['legal_scarcity'])))
                validation['a1_scarcity_prediction_difference']=max(validation['a1_scarcity_prediction_difference'],delta)
        if si % 10 == 0 or si+1 == len(paths):
            print(f'Evaluation and uncertainty: {si+1}/{len(paths)} shards ({time.monotonic()-started:.1f}s)',flush=True)
    output_stages=[]; coeff_rows=[]; model_rows=[]; enrichment_rows=[]; contrasts=[]
    for stage,label in enumerate(LABELS):
        ntrain=sums[stage]['n']; target=np.concatenate(targets[stage]); result=dict(label=label,train_boards=ntrain,test_boards=len(target),models={})
        assert ntrain == manifest['config']['train_boards']
        assert len(target) == manifest['config']['test_boards']
        for key,fit in fits[stage].items():
            error=np.concatenate(fit['errors']); hits=np.concatenate(fit['hits']); gaps=np.concatenate(fit['gaps'])
            fit['errors']=error
            r2,r2ci=r2_stats(error,target); agreement,agreeci=interval_mean(hits)
            score_mean=fit['score_sum']/ntrain
            validation['max_training_score_mean']=max(validation['max_training_score_mean'],float(np.max(np.abs(score_mean),initial=0)))
            a=fit['active']
            cov=score_covariance(fit['score_sum'],fit['score_outer'],fit['inverse'],fit['scale'][a],ntrain)
            fullcov=np.zeros((len(fit['raw']),len(fit['raw']))); fullcov[np.ix_(a,a)]=cov
            fit['covariance']=fullcov
            coefficients=[]
            for j,feature in enumerate(specs[key]):
                estimated=j in fit['active_indices']; se=float(np.sqrt(max(0,fullcov[j,j]))) if estimated else None
                value=float(fit['raw'][j]) if estimated else None
                row=dict(stage=label,model=key,feature=feature,coefficient=value,standard_error=se,
                         ci95_low=value-Z95*se if estimated else None,ci95_high=value+Z95*se if estimated else None,
                         standardized_coefficient=float(fit['standardized'][j]) if estimated else None,
                         status='estimated' if estimated else 'omitted_constant_or_exact_dependency')
                if 'scarcity_interaction' in feature:
                    row.update(family40_ci_low=value-Z40*se if estimated else None,family40_ci_high=value+Z40*se if estimated else None)
                coefficients.append(row); coeff_rows.append(row)
            res=dict(r2=r2,r2_ci95=r2ci,rmse=float(np.sqrt(error.mean())),top_choice_agreement=agreement,
                     top_choice_agreement_ci95=agreeci,mean_logit_gap=float(gaps.mean()),coefficients=coefficients)
            result['models'][key]=res
            model_rows.append(dict(stage=label,model=key,r2=r2,r2_ci_low=r2ci[0],r2_ci_high=r2ci[1],
                                   top_choice_agreement=agreement,test_boards=len(target)))
        comparisons=[('without_'+r,'resource_pips') for r in RESOURCE]+[('resource_pips','resource_coverage'),('resource_coverage','resource_coverage_ports'),
                     ('resource_coverage_ports','resource_coverage_ports_numbers'),('resource_coverage_ports_numbers','full_context'),
                     ('full_context','resource_context'),('resource_context','board_scarcity'),('resource_context','legal_scarcity')]
        result['paired_model_comparisons']=[dict(reduced=a,full=b,**paired_r2_difference(fits[stage][a]['errors'],fits[stage][b]['errors'],target)) for a,b in comparisons]
        f=fits[stage]['resource_pips']
        for i in range(5):
            for j in range(i):
                val=float(f['raw'][i]-f['raw'][j]); se=float(np.sqrt(max(0,f['covariance'][i,i]+f['covariance'][j,j]-2*f['covariance'][i,j])))
                contrasts.append(dict(stage=label,resource_added=RESOURCE[i],resource_replaced=RESOURCE[j],logit_change_per_pip=val,
                                      ci95=[val-Z95*se,val+Z95*se],family40_ci=[val-Z40*se,val+Z40*se]))
        raw={key:np.concatenate([item[key] for item in descriptive[stage]]) for key in descriptive[stage][0]}
        np.savez_compressed(destination/f'descriptive-{label}.npz',**raw)
        enriched={}
        for baseline_name in ['baseline','matched_baseline']:
            eligible=np.ones(len(raw['seed']),dtype=bool) if baseline_name=='baseline' else raw['matched_count']>=3
            allstats=enriched_stats(raw['chosen'][eligible],raw[baseline_name][eligible])
            enriched[baseline_name]=dict(eligible_boards=int(eligible.sum()),total_boards=len(eligible),statistics=allstats,by_existing_access={})
            if baseline_name == 'matched_baseline':
                enriched[baseline_name]['matching_pool_size_quantiles'] = np.quantile(raw['matched_count'][eligible],[0,.25,.5,.75,1]).tolist() if eligible.any() else []
            for j,r in enumerate(RESOURCE):
                if allstats['n']:
                    enrichment_rows.append(dict(stage=label,baseline=baseline_name,resource=r,boards=allstats['n'],
                        chosen_share=allstats['chosen_share'][j],baseline_share=allstats['baseline_share'][j],enrichment=allstats['ratio_of_means'][j],
                        enrichment_ci_low=allstats['ratio_ci95'][j][0] if allstats['ratio_ci95'][j] else None,
                        enrichment_ci_high=allstats['ratio_ci95'][j][1] if allstats['ratio_ci95'][j] else None))
                access={}
                for condition in ['missing','already_owned']:
                    select=eligible & ((raw['own'][:,j]==0) if condition=='missing' else (raw['own'][:,j]>0))
                    stats=enriched_stats(raw['chosen'][select],raw[baseline_name][select])
                    access[condition]=dict(boards=stats['n']) if not stats['n'] else dict(boards=stats['n'],
                        chosen_share=stats['chosen_share'][j],baseline_share=stats['baseline_share'][j],
                        enrichment=stats['ratio_of_means'][j],ci95=stats['ratio_ci95'][j])
                enriched[baseline_name]['by_existing_access'][r]=access
        result['enrichment']=enriched
        result['availability']=dict(mean_board_pips=raw['supply'].mean(0).tolist(),mean_legal_candidate_pips=raw['legal_mean'].mean(0).tolist(),
                                    excluded_zero_pip_candidate_count=int(raw['zero_pip_candidates'].sum()))
        result['context_support'] = {r:dict(missing_resource_boards=int((raw['own'][:,j]==0).sum()),
                already_owned_resource_boards=int((raw['own'][:,j]>0).sum()),
                owned_matching_port_boards=int((raw['owned_port'][:,j]>0).sum())) for j,r in enumerate(RESOURCE)}
        output_stages.append(result)
    assert validation['a1_scarcity_prediction_difference']<1e-5
    assert validation['max_training_score_mean']<1e-6
    report=dict(schema='opening-scale-analysis-v1',source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                boards=manifest['config']['boards'],model=manifest['model'],resources=RESOURCE,stages=output_stages,
                resource_substitution_contrasts=contrasts,feature_union=union_names,model_definitions=specs,validation=validation,
                method=dict(target='Within-decision-centered legal settlement logit, not win probability.',
                    fitting='Separate fits by stage. Each training board has equal weight and candidates share its weight. Training-RMS standardized ridge alpha=0.001; exact dependencies omitted in feature order.',
                    coefficients='Raw logit units per feature unit; all five base resource coefficients share pip units.',
                    coefficient_uncertainty='Board-cluster sandwich around the fixed-ridge estimand, including empirical RMS scaling; retained feature set held fixed. Normal 95% intervals.',
                    test_uncertainty='Whole-board delta-method intervals for held-out R² and paired ΔR²; conditional on fitted coefficients.',
                    enrichment='Ratio of board-equal mean chosen resource fraction to mean baseline fraction, across all generated boards. Log-ratio delta-method intervals, paired by board. Not mean of per-board ratios.',
                    baseline='Uniformly random currently legal positive-production corner. Zero-pip corners excluded only from this descriptive resource-share baseline.',
                    matched_baseline='Uniform among legal corners with the chosen corner\'s exact total pips and adjacent-hex count (coastal geometry), including the chosen corner; report only boards with at least 3 matching corners.',
                    board_scarcity='P_r(candidate) × (58 × tile_count_r / 18 − total_board_resource_pips_r). Positive coefficient associates lower supply with greater per-pip priority.',
                    legal_scarcity='P_r(candidate) × (58 × tile_count_r / 162 − mean_legal_corner_resource_pips_r). Separate alternative model; at A1 exactly board_scarcity/9.',
                    scarcity_multiple_comparisons='Also report Bonferroni intervals for the 40 resource×stage×availability-measure coefficients; A1 measures are equivalent, so this is conservative.',
                    resource_context='Full original context plus each resource\'s candidate-pip interactions with owned pips, missing access, owned matching 2:1 port, geometric port proximity; resource-specific direct ports and starting cards.'),
                limitations=['Frozen Generation 37 network preferences under one greedy opening trajectory per random board; no search, full-game outcomes, or causal intervention.',
                    'Generator-distribution conclusions; later-stage histories are conditional on preceding greedy choices, not representative of all human openings.',
                    'More boards reduce sampling uncertainty, not confounding, model misspecification, policy bias, or dependence on the trained checkpoint.',
                    'Matched baselines condition on chosen total production and geometry, and cover an explicitly reported subset.',
                    'Scarcity regressions describe conditional associations, not causal resource values. Board supply and legal availability are correlated; they are fitted separately.',
                    'Intervals condition on the analysis specification and model training; no uncertainty over independently trained policies is included.'],
                elapsed_seconds=round(time.monotonic()-started,3))
    atomic_json(destination/'results.json',report)
    for filename,rows in [('coefficients.csv',coeff_rows),('models.csv',model_rows),('enrichment.csv',enrichment_rows)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (destination/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(boards=report['boards'],elapsed_seconds=report['elapsed_seconds'],validation=validation),indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('run/analysis/opening-study-100k-20261029'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    analyze(args.source,args.out or args.source/'analysis')


if __name__=='__main__':
    main()
