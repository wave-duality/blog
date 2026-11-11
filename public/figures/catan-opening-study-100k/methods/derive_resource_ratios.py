#!/usr/bin/env python3
"""Derive additive-surrogate pip ratios without refitting or network inference.

The existing analysis exports marginal SEs and all pairwise difference CIs.
Together they identify the complete five-coefficient board-cluster covariance.
Requires Python 3 and NumPy. To run from any directory:
  python derive_resource_ratios.py --source /path/to/results.json --output /path/to/resource-ratios.json
The defaults also work when the original public folder structure is retained:
  results.json, resource-ratios.json, methods/derive_resource_ratios.py
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=HERE.parent/'results.json',
                    help='Original study results.json; defaults to ../results.json beside the public methods/ directory.')
parser.add_argument('--output', type=Path, default=HERE.parent/'resource-ratios.json',
                    help='Destination JSON (default: ../resource-ratios.json).')
args = parser.parse_args()
SOURCE = args.source.resolve()
DESTINATION = args.output.resolve()
if SOURCE == DESTINATION:
    parser.error('--output must differ from --source')
Z = NormalDist().inv_cdf(.975)
resources = ['wood', 'brick', 'sheep', 'wheat', 'ore']
data = json.loads(SOURCE.read_text())
output = {
    'source_filename': SOURCE.name,
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'method': {
        'model': 'resource_pips (five-resource additive logit surrogate)',
        'ratio': 'beta_numerator / beta_denominator; denominator-resource production pips per numerator-resource production pip at equal fitted score',
        'covariance': 'Use marginal SEs and Var(beta_i - beta_j) from published symmetric normal 95% contrast intervals: Cov(i,j) = (Var(i)+Var(j)-Var(i-j))/2.',
        'interval': 'Pointwise 95% log-ratio delta-method interval, exp(log(ratio) +/- z_0.975 * sqrt(g^T Cov(beta) g)), with g_i=1/beta_i and g_j=-1/beta_j.',
        'sampling_unit': 'Whole training board; covariance inherited from the original board-cluster sandwich, including empirical RMS scaling, around the fixed-ridge estimand.',
        'caveats': [
            'Production pips, not resource cards in hand or bank-trade exchange rates.',
            'Describes the fitted additive score surrogate, not a causal intervention or winning-probability equivalence.',
            'Ratios allow total production to change. A fixed-total-pip resource substitution has fitted score change beta_i-beta_j instead.',
            'Geometry and other omitted features may correlate with resource pips. The five-pip regression does not hold them fixed.',
            'These intervals measure board-sampling uncertainty conditional on this frozen checkpoint and fixed analysis specification, not model misspecification or training-run uncertainty.',
            'Intervals are pointwise; no simultaneous multiple-comparison guarantee.',
            'All coefficients are positive and their estimates far from zero, making log-delta intervals appropriate here.',
        ],
    },
    'resources': resources,
    'stages': [],
}
for stage in data['stages']:
    label = stage['label']
    model = stage['models']['resource_pips']
    coeff = {c['feature']: c for c in model['coefficients']}
    beta = np.array([coeff[r+'_pips']['coefficient'] for r in resources])
    marginal_se = np.array([coeff[r+'_pips']['standard_error'] for r in resources])
    cov = np.diag(marginal_se**2)
    contrasts = [c for c in data['resource_substitution_contrasts'] if c['stage']==label]
    assert len(contrasts)==10
    for c in contrasts:
        i = resources.index(c['resource_added'])
        j = resources.index(c['resource_replaced'])
        se = (c['ci95'][1]-c['ci95'][0])/(2*Z)
        assert abs(beta[i]-beta[j]-c['logit_change_per_pip']) < 1e-12
        cov[i,j] = cov[j,i] = (cov[i,i]+cov[j,j]-se**2)/2
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > 0
    assert np.all(beta-10*marginal_se > 0)
    ratios=[]
    for i,numerator in enumerate(resources):
        for j,denominator in enumerate(resources):
            ratio = beta[i]/beta[j]
            g = np.zeros(5)
            g[i] += 1/beta[i]
            g[j] -= 1/beta[j]
            log_se = math.sqrt(max(0,float(g@cov@g)))
            ci = [ratio*math.exp(-Z*log_se),ratio*math.exp(Z*log_se)]
            difference = beta[i]-beta[j]
            diff_se = math.sqrt(max(0,float(cov[i,i]+cov[j,j]-2*cov[i,j])))
            ratios.append({'numerator':numerator,'denominator':denominator,
                'ratio':float(ratio),'log_ratio_standard_error':log_se,'ci95':ci,
                'fixed_total_substitution_logit_difference':float(difference),
                'fixed_total_substitution_difference_ci95':[difference-Z*diff_se,difference+Z*diff_se],
                'implied_softmax_odds_multiplier_for_substitution':math.exp(difference)})
    output['stages'].append({'stage':label,'train_boards':stage['train_boards'],
        'held_out_r2':model['r2'],'raw_coefficients':beta.tolist(),
        'coefficient_covariance':cov.tolist(),'covariance_eigenvalues':eigenvalues.tolist(),
        'ratios':ratios})
    print(label, 'sheep=1:', ', '.join(f"{r['numerator']} {r['ratio']:.6f} [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]" for r in ratios if r['denominator']=='sheep'))
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
DESTINATION.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print('Saved', DESTINATION)
