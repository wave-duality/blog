#!/usr/bin/env python3
"""Frozen Gen37 opening study: direct CPU policy, all legal candidates, no search.

Each board has one greedy A1/B1/B2/A2 history. The explanatory ridge regressions
are fitted to relative legal logits; they do not modify or train the policy.
Reproduce: .venv/bin/python opening_network_study.py
Re-fit saved data: .venv/bin/python opening_network_study.py --refit
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import catan_rl
import numpy as np
import torch

from train import split_obs
from opening_engine import OpeningEngine, GEOMETRY, LABELS, SEATS, _from_arrays, _encode
from feature_dictionary import geometry_data, ownership
from analyze_number_diversification import production, PIPS

ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = ROOT / 'run/analysis/opening-study-20261028'
RESOURCE = ['wood', 'brick', 'sheep', 'wheat', 'ore']
SPLIT_SEED = 281037
ALPHA = .001
SCHEMA = 'opening-network-study-v1'


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def board_features(board, history, seat, geometry, distance):
    """All vertices; raw units. Distance is geometric, not affordable travel."""
    own, opponent = ownership(history, seat)
    yields = np.array([production(board, [v], geometry) for v in range(54)])
    rp = np.einsum('n,vnr->vr', PIPS, yields)
    owned_yields = production(board, own, geometry)
    owned_rp = PIPS @ owned_yields
    combined = rp + owned_rp
    total = rp.sum(1)
    combined_total = combined.sum(1)
    shares = np.divide(combined, combined_total[:, None], out=np.zeros_like(combined, dtype=float), where=combined_total[:, None] > 0)
    port = np.full(54, -1, dtype=int)
    port_distances = np.empty((54, 6), dtype=int)
    for kind, vertices in zip(board[2], geometry['port_vertices']):
        port[vertices] = kind
    for r in range(6):
        port_distances[:, r] = distance[:, np.flatnonzero(port == r)].min(1)
    owned_ports = set(port[own]) - {-1}
    effective_distance = port_distances.copy()
    for r in owned_ports:
        effective_distance[:, r] = 0
    matched = np.array([combined[v, port[v]] if 0 <= port[v] < 5 else 0 for v in range(54)])
    counts = yields.sum(2)
    combined_counts = counts + owned_yields.sum(1)
    dice_probability = PIPS / 36
    dice_probability = dice_probability.copy(); dice_probability[7] = 6/36
    occupied = own + opponent
    open_mask = np.ones(54, dtype=bool)
    if occupied:
        open_mask = distance[:, occupied].min(1) > 1
    expansion = np.array([total[(distance[v] == 2) & open_mask].max(initial=0) for v in range(54)])
    values = {}; descriptions = {}

    def add(name, value, description):
        values[name] = np.asarray(value, dtype=float)
        descriptions[name] = description

    add('total_pips', total, 'Expected production numerator: cards per roll = pips / 36, ignoring robber and bank shortage.')
    for r, name in enumerate(RESOURCE):
        add(name + '_pips', rp[:, r], 'Candidate settlement production pips for ' + name + '.')
    add('adjacent_hexes', [sum(h is not None and h >= 0 for h in hs) for hs in geometry['vertex_hexes']], 'Number of touching hexes, including desert; coastal/interior control.')
    add('local_resource_count', (rp > 0).sum(1), 'Distinct producing resources touched by this candidate.')
    add('new_resource_types', ((rp > 0) & (owned_rp == 0)).sum(1), 'Resources added that existing own settlements do not produce.')
    add('new_resource_pips', (rp * (owned_rp == 0)).sum(1), 'Candidate pips in resources missing from existing own settlements.')
    add('combined_resource_count', (combined > 0).sum(1), 'Distinct resources after adding this candidate to existing own settlements.')
    add('all_five_covered', (combined > 0).all(1), 'One if the combined own portfolio produces all five resources.')
    add('combined_total_pips', combined_total, 'Sum of candidate and existing own settlement pips, counting each settlement.')
    add('combined_concentration_hhi', (shares * shares).sum(1), 'Sum of squared resource shares; 1 means one-resource concentration, 0.2 means five equal shares.')
    add('combined_resource_entropy', -(shares * np.log(np.maximum(shares, 1e-300))).sum(1), 'Natural-log entropy of combined resource pip shares; zero when concentrated in one resource.')
    for r, name in enumerate(RESOURCE):
        add('combined_' + name + '_pips', combined[:, r], 'Own portfolio production pips after adding the candidate for ' + name + '.')
        add(name + '_starting_cards', yields[:, :, r].sum(1) if len(own) else np.zeros(54), 'Cards of this resource granted immediately if this is the actor\'s second settlement; first placements receive zero.')
    add('direct_port', port >= 0, 'One if the candidate touches any port.')
    add('generic_port', port == 5, 'One if the candidate touches a generic 3:1 port.')
    for r, name in enumerate(RESOURCE):
        add(name + '_port', port == r, 'One if the candidate touches the ' + name + ' 2:1 port.')
    add('port_matching_combined_pips', matched, 'Combined own pips in the specialized port resource, zero unless this candidate directly touches that port.')
    add('nearest_port_distance', port_distances.min(1), 'Shortest geometric edge distance from this candidate to any port corner; ignores ownership, road direction, cost and blocking.')
    add('nearest_generic_port_distance', port_distances[:, 5], 'Shortest geometric edge distance to any generic 3:1 port corner.')
    for r, name in enumerate(RESOURCE):
        add(name + '_port_distance', port_distances[:, r], 'Shortest geometric edge distance to the ' + name + ' port.')
    add('matched_port_potential', (combined / (1 + effective_distance[:, :5])).max(1), 'Maximum over resources of combined resource pips / (1 + geometric port distance); already owned ports use distance zero. A descriptive interaction, not a trade-value model.')
    add('unique_local_numbers', (counts > 0).sum(1), 'Distinct dice outcomes producing cards at this candidate.')
    add('unique_combined_numbers', (combined_counts > 0).sum(1), 'Distinct dice outcomes producing cards for the combined own portfolio.')
    add('pips_overlapping_owned_numbers', (counts * (owned_yields.sum(1) > 0) * PIPS).sum(1), 'Candidate pips on number outcomes already collected by an own settlement.')
    add('combined_production_variance', (combined_counts**2) @ dice_probability - (combined_counts @ dice_probability)**2, 'Variance in total cards per roll, counting shared numbers exactly; ignores robber and bank shortage.')
    add('combined_zero_production_probability', (combined_counts == 0) @ dice_probability, 'Probability of collecting zero cards on a dice roll, including seven; ignores robber and bank shortage.')
    add('best_expansion_pips_two_roads', expansion, 'Highest-pip currently distance-rule-open corner exactly two geometric edges away; excludes cost, direction and competition.')
    add('opponent_proximity', 1/(1+distance[:, opponent].min(1)) if opponent else np.zeros(54), '1/(1 + closest opponent settlement edge distance); zero if no opponent settlement yet.')
    add('own_proximity', 1/(1+distance[:, own].min(1)) if own else np.zeros(54), '1/(1 + closest own settlement edge distance); zero if no own settlement yet.')
    add('desert_adjacency', [any(h is not None and h >= 0 and board[0][h] == 5 for h in hs) for hs in geometry['vertex_hexes']], 'One if this candidate touches the desert.')
    return list(values), np.column_stack(list(values.values())), descriptions


def feature_models():
    resources = [r + '_pips' for r in RESOURCE]
    coverage = resources + ['adjacent_hexes', 'local_resource_count', 'new_resource_types', 'new_resource_pips', 'all_five_covered', 'combined_concentration_hhi']
    ports = coverage + ['direct_port', 'generic_port', 'port_matching_combined_pips', 'nearest_port_distance', 'nearest_generic_port_distance', 'matched_port_potential']
    full = ports + ['unique_local_numbers', 'unique_combined_numbers', 'pips_overlapping_owned_numbers', 'combined_production_variance', 'best_expansion_pips_two_roads', 'opponent_proximity', 'own_proximity']
    return dict(total_pips=['total_pips'], resource_pips=resources, resource_coverage=coverage,
                resource_coverage_ports=ports, full_context=full)


def code_for(board, history):
    return _encode(_from_arrays(*board, [a for _, a in history]))


@torch.inference_mode()
def build_board(engine, seed, split, geometry, distance):
    game = catan_rl.Game(seed=int(seed))
    board = [list(x) for x in game.board_arrays()]
    picked = []
    captured = {}
    hook = engine.model.settle_head.register_forward_pre_hook(lambda module, args: captured.update(hidden=args[0][0]))
    weights = engine.model.settle_head.weight[0].numpy().astype(float)
    try:
        for stage, label in enumerate(LABELS):
            history = [list(x) for x in game.history()]
            assert game.to_act() == SEATS[stage]
            legal = np.array(game.legal(), dtype=int)
            assert np.all((legal >= 72) & (legal < 126))
            logits, _ = engine.model(*split_obs(torch.from_numpy(game.opening_observation())))
            scores = logits[0, legal].numpy().astype(float)
            vertices = legal - 72
            order = np.lexsort((legal, -scores))
            winner, second = int(order[0]), int(order[1])
            chosen, runner_up = int(vertices[winner]), int(vertices[second])
            probability = np.exp(scores - scores.max()); probability /= probability.sum()
            names, features, descriptions = board_features(board, history, SEATS[stage], geometry, distance)
            hidden = captured['hidden'].numpy().astype(float)
            terms = weights * (hidden[chosen] - hidden[runner_up])
            margin = float(scores[winner] - scores[second])
            reconstruction_error = abs(float(terms.sum()) - margin)
            assert reconstruction_error < 1e-4, reconstruction_error
            channels = np.argsort(-np.abs(terms))[:12]
            head = [dict(channel=int(i), chosen_activation=float(hidden[chosen, i]), alternative_activation=float(hidden[runner_up, i]), weight=float(weights[i]), contribution=float(terms[i])) for i in channels]
            candidates = [dict(vertex=int(v), logit=float(z), preference=float(p), chosen=bool(v == chosen), features=features[v].tolist()) for v, z, p in zip(vertices, scores, probability)]
            chosen_features = dict(zip(names, features[chosen].tolist()))
            pips = features[vertices, names.index('total_pips')]
            game.apply(chosen + 72)
            road_legal = np.array(game.legal(), dtype=int)
            road_logits, _ = engine.model(*split_obs(torch.from_numpy(game.opening_observation())))
            road_scores = road_logits[0, road_legal].numpy().astype(float)
            road_order = np.lexsort((road_legal, -road_scores))
            road = int(road_legal[road_order[0]])
            road_probs = np.exp(road_scores - road_scores.max()); road_probs /= road_probs.sum()
            game.apply(road)
            picked.append(dict(label=label, seat=int(SEATS[stage]), history=history, code=code_for(board, history), chosen=chosen, road=road,
                               runner_up=runner_up, margin=margin, candidates=candidates, chosen_features=chosen_features,
                               legal_count=len(candidates), legal_min_pips=float(pips.min()), legal_max_pips=float(pips.max()),
                               pip_sacrifice=float(pips.max() - chosen_features['total_pips']),
                               head_contributions=head, head_other=float(terms.sum() - terms[channels].sum()), head_reconstruction_error=reconstruction_error,
                               road_candidates=[dict(road=int(a), logit=float(z), preference=float(p)) for a, z, p in zip(road_legal, road_scores, road_probs)]))
    finally:
        hook.remove()
    return dict(schema=SCHEMA, model_sha256=engine.metadata['sha256'], seed=int(seed), split=split, board=board, code=code_for(board, []), history=[list(x) for x in game.history()], picks=picked), names, descriptions


def centered(x, group):
    out = np.array(x, dtype=float, copy=True)
    for g in np.unique(group):
        mask = group == g; out[mask] -= out[mask].mean(axis=0)
    return out


def regress(x, y, groups, seeds, vertices, train_mask, columns, names):
    """Fixed standardized ridge, position-balanced squared error; no test tuning."""
    selected = np.array([names.index(c) for c in columns])
    xx = x[:, selected]
    _, reverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1/counts[reverse]
    train_weight = weight * train_mask
    train_weight /= train_weight.sum()
    scale = np.sqrt((xx**2 * train_weight[:, None]).sum(0))
    active = scale > 1e-10
    scale[~active] = 1
    scaled = xx / scale
    covariance = scaled.T @ (scaled * train_weight[:, None])
    rhs = scaled.T @ (y * train_weight)
    # Preserve feature order, but omit exact dependencies using training data
    # only. In A1/B1, e.g. new-resource pips equals the sum of resource pips.
    independent = []; omitted = []
    for j, name in enumerate(columns):
        trial = independent + [j]
        if active[j] and np.linalg.matrix_rank(covariance[np.ix_(trial, trial)], tol=1e-9) == len(trial):
            independent.append(j)
        else:
            omitted.append(name)
    beta = np.zeros(len(columns))
    if independent:
        beta[independent] = np.linalg.solve(covariance[np.ix_(independent, independent)] + ALPHA*np.eye(len(independent)), rhs[independent])
    prediction = scaled @ beta
    test = ~train_mask
    test_weight = weight[test]; test_weight /= test_weight.sum()
    residual = y[test] - prediction[test]
    mse = float(test_weight @ residual**2)
    baseline = float(test_weight @ y[test]**2)
    agreement = []; gaps = []; per_board = []; choices = {}
    for g in np.unique(groups[test]):
        rows = np.flatnonzero(groups == g)
        actual = rows[np.lexsort((vertices[rows], -y[rows]))[0]]
        estimated = rows[np.lexsort((vertices[rows], -prediction[rows]))[0]]
        hit = bool(vertices[estimated] == vertices[actual])
        gap = float(y[actual] - y[estimated])
        agreement.append(hit); gaps.append(gap)
        per_board.append((float(np.mean((y[rows]-prediction[rows])**2)), float(np.mean(y[rows]**2)), int(hit), gap))
        choices[int(g)] = dict(chosen=int(vertices[estimated]), agrees=hit, network_logit_gap=gap)
    # One independent test decision per board at each stage.
    samples = np.asarray(per_board)
    rng = np.random.default_rng(280037)
    boot = samples[rng.integers(len(samples), size=(1000, len(samples)))].mean(1)
    r2_ci = np.quantile(1-boot[:, 0]/boot[:, 1], [.025, .975]).tolist()
    agreement_ci = np.quantile(boot[:, 2], [.025, .975]).tolist()
    return dict(features=columns, ridge_alpha=ALPHA, coefficients=(beta/scale).tolist(), standardized_coefficients=beta.tolist(), training_rms_scale=scale.tolist(),
                r2=float(1-mse/baseline), r2_board_bootstrap_95=r2_ci, rmse=float(np.sqrt(mse)), top_choice_agreement=float(np.mean(agreement)),
                top_choice_agreement_board_bootstrap_95=agreement_ci, mean_network_logit_gap=float(np.mean(gaps)), test_boards=len(agreement),
                inactive_constant_features=[c for c, a in zip(columns, active) if not a], omitted_constant_or_redundant_features=omitted), choices, prediction


def analyze(dest, manifest):
    names = manifest['feature_names']
    boards = [json.loads((dest / row['path']).read_text()) for row in manifest['boards']]
    xx=[]; yy=[]; groups=[]; seed_rows=[]; vertices=[]; stages=[]; train=[]
    for bi, board in enumerate(boards):
        for si, pick in enumerate(board['picks']):
            for candidate in pick['candidates']:
                xx.append(candidate['features']); yy.append(candidate['logit']); groups.append(bi*4+si)
                seed_rows.append(board['seed']); vertices.append(candidate['vertex']); stages.append(si); train.append(board['split']=='train')
    arrays = dict(features=np.asarray(xx, dtype=np.float64), logits=np.asarray(yy, dtype=np.float64), position=np.asarray(groups, dtype=np.int32),
                  seed=np.asarray(seed_rows, dtype=np.int64), vertex=np.asarray(vertices, dtype=np.int16), stage=np.asarray(stages, dtype=np.int8), train=np.asarray(train, dtype=bool))
    x = centered(arrays['features'], arrays['position']); y = centered(arrays['logits'], arrays['position'])
    np.savez_compressed(dest/'study_arrays.npz', **arrays)
    model_sets = feature_models()
    results = []; recommendations = {}
    for si, label in enumerate(LABELS):
        mask = arrays['stage'] == si
        summary = dict(label=label, seat=int(SEATS[si]), train_boards=int(len(np.unique(arrays['seed'][mask & arrays['train']]))),
                       test_boards=int(len(np.unique(arrays['seed'][mask & ~arrays['train']]))), candidates=int(mask.sum()), models={}, single_features={})
        for key, columns in model_sets.items():
            fit, choices, prediction = regress(x[mask], y[mask], arrays['position'][mask], arrays['seed'][mask], arrays['vertex'][mask], arrays['train'][mask], columns, names)
            summary['models'][key] = fit
            for group, choice in choices.items():
                recommendations.setdefault(group, {})[key] = choice
            # Save all fitted candidate predictions, with provenance by board split.
            arrays['prediction_' + key + '_' + label] = prediction
        for feature in ['total_pips', *[r+'_pips' for r in RESOURCE], 'new_resource_types', 'all_five_covered', 'nearest_port_distance', 'port_matching_combined_pips', 'combined_concentration_hhi']:
            fit, _, _ = regress(x[mask], y[mask], arrays['position'][mask], arrays['seed'][mask], arrays['vertex'][mask], arrays['train'][mask], [feature], names)
            summary['single_features'][feature] = fit
        chosen = [b['picks'][si] for b in boards if b['split']=='test']
        summary['chosen_on_test'] = dict(mean_pips=float(np.mean([p['chosen_features']['total_pips'] for p in chosen])),
            mean_pip_sacrifice=float(np.mean([p['pip_sacrifice'] for p in chosen])), max_pips_choice_rate=float(np.mean([p['pip_sacrifice']==0 for p in chosen])),
            mean_margin=float(np.mean([p['margin'] for p in chosen])), combined_coverage_histogram=dict(sorted(Counter(int(p['chosen_features']['combined_resource_count']) for p in chosen).items())),
            direct_port_rate=float(np.mean([p['chosen_features']['direct_port'] for p in chosen])))
        results.append(summary)
    np.savez_compressed(dest/'study_arrays.npz', **arrays)
    for bi, board in enumerate(boards):
        if board['split'] == 'test':
            for si, pick in enumerate(board['picks']):
                pick['model_predictions'] = recommendations[bi*4+si]
            json_write(dest/'boards'/f"{board['seed']}.json", board)
    metrics = dict(schema=SCHEMA, model=manifest['model'], split=manifest['split'], ridge_alpha=ALPHA, stages=results,
                   response='Within-decision centered raw legal settlement logit; features centered over the same legal candidates.',
                   weighting='Each decision has equal weight; candidates within a decision share its weight equally. Models fitted independently by A1/B1/B2/A2.',
                   r2_definition='1 - mean_test_boards(mean_candidates((centered_logit - prediction)^2)) / mean_test_boards(mean_candidates(centered_logit^2)). No intercept after centering.',
                   uncertainty='1000 percentile bootstrap resamples of the 200 test boards, separately by stage; intervals condition on the fitted model and do not include training-set variability.',
                   selection_metric='Exact agreement with deterministic highest-legal-logit choice; lower vertex index breaks ties.',
                   interpretation='Descriptive fidelity to the frozen network policy, not win prediction, causal feature importance, searched play, or optimal play.')
    json_write(dest/'metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boards', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=3100000)
    parser.add_argument('--out', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args(); dest = args.out
    if args.refit:
        manifest = json.loads((dest/'manifest.json').read_text()); analyze(dest, manifest); return
    if args.boards < 5:
        raise ValueError('At least five boards are required for the 80/20 split.')
    dest.mkdir(parents=True, exist_ok=True); (dest/'boards').mkdir(exist_ok=True)
    seeds = np.arange(args.seed, args.seed + args.boards)
    previous = json.loads((ROOT/'run/analysis/opening-gallery-20261004/manifest.json').read_text())
    assert not set(seeds) & {r['seed'] for r in previous['boards']}
    assert not set(seeds) & set(range(2040000, 2040256)), 'Reserved confirmation seeds'
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(seeds)
    train_seeds = set(shuffled[:4*args.boards//5])
    engine = OpeningEngine(); geometry, distance = geometry_data()
    started = time.monotonic(); rows = []
    manifest = dict(schema=SCHEMA, status='running', board_count=args.boards, target_count=args.boards, completed_count=0, labels=list(LABELS), seats=list(SEATS), model=engine.metadata,
                    seed_start=args.seed, seed_end=args.seed+args.boards-1, split=dict(method='Fixed shuffled whole-board 80/20 split', rng_seed=SPLIT_SEED, train_boards=len(train_seeds), test_boards=args.boards-len(train_seeds)),
                    selection='All eight setup actions are greedy raw-policy argmax; one conditional trajectory per generated board.',
                    history_format='[[1, action_id], ...]; settlement action 72 + vertex, road action road index; owners follow A1/B1/B2/A2.',
                    geometry=geometry, feature_names=[], feature_descriptions={}, boards=[], metrics_path='metrics.json',
                    candidate_features='Candidate features are raw-unit arrays indexed by feature_names.',
                    source_script='opening_network_study.py', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limitations=['Random boards from this engine generator, not uniform across all conceivable Catan boards.',
                                 'One greedy prefix per board; A2 and B2 observations are conditional on earlier greedy choices.',
                                 'Softmax preferences are neither calibrated win probabilities nor sampling frequencies.',
                                 'Distances are geometric and do not measure build cost, legal path availability or affordable port access.',
                                 'Final-head contributions exactly account for a chosen-versus-runner-up logit margin; channels have no established causal strategic meaning.'])
    for index, seed in enumerate(seeds):
        split = 'train' if seed in train_seeds else 'test'
        path = dest/'boards'/f'{seed}.json'
        if path.exists():
            board = json.loads(path.read_text())
            assert board['seed'] == seed and board['split'] == split and board['schema'] == SCHEMA and board['model_sha256'] == engine.metadata['sha256']
            names, _, descriptions = board_features(board['board'], [], 0, geometry, distance)
        else:
            board, names, descriptions = build_board(engine, int(seed), split, geometry, distance)
            json_write(path, board)
        rows.append(dict(seed=int(seed), split=split, path=f'boards/{seed}.json', code=board['code'],
                         picks=[{key:p[key] for key in ['label','seat','chosen','road','runner_up','margin','legal_count','legal_min_pips','legal_max_pips','pip_sacrifice','chosen_features']} for p in board['picks']]))
        manifest.update(completed_count=index+1, feature_names=names, feature_descriptions=descriptions, boards=rows)
        if (index+1) % 25 == 0 or index == 0 or index+1 == len(seeds):
            json_write(dest/'manifest.json', manifest)
            print(f'{index+1}/{len(seeds)} boards; {8*(index+1)} direct forwards; {time.monotonic()-started:.1f}s', flush=True)
    manifest.update(status='fitting', forward_passes=8*len(seeds), generation_elapsed_seconds=round(time.monotonic()-started, 3), model_names=list(feature_models()))
    json_write(dest/'manifest.json', manifest)
    metrics = analyze(dest, manifest)
    manifest.update(status='complete', elapsed_seconds=round(time.monotonic()-started, 3), candidate_count=sum(s['candidates'] for s in metrics['stages']))
    json_write(dest/'manifest.json', manifest)
    print(json.dumps({s['label']:{k:dict(r2=round(v['r2'],4),agreement=round(v['top_choice_agreement'],4)) for k,v in s['models'].items()} for s in metrics['stages']}, indent=2), flush=True)
    print(dest, flush=True)


if __name__ == '__main__':
    main()
