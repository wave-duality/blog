#!/usr/bin/env python3
"""Render a self-contained report from the completed opening-scale analysis.

This reads saved statistics only. It never runs the policy, fits a regression,
changes the article, or uses pilot results to select a scientific narrative.
"""
import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'run/analysis/opening-study-100k-20261029/analysis'
RESOURCES = ['wood', 'brick', 'sheep', 'wheat', 'ore']
STAGES = ['A1', 'B1', 'B2', 'A2']
COLORS = dict(wood='#286754', brick='#b65235', sheep='#718f37', wheat='#b28615', ore='#586b88')
MODEL_LABELS = {
    'total_pips': 'Total pips (one shared weight)',
    'resource_pips': 'Five resource pip counts',
    'resource_coverage': '+ Coverage, concentration, adjacent tiles',
    'resource_coverage_ports': '+ Port features',
    'resource_coverage_ports_numbers': '+ Dice-number features',
    'full_context': '+ Expansion and settlement proximity',
    'resource_context': '+ Resource-specific context',
    'board_scarcity': '+ Board-supply interactions',
    'legal_scarcity': '+ Legal-availability interactions (alternative)',
}


def number(value, digits=3, signed=False):
    if value is None or not math.isfinite(float(value)):
        return '—'
    return format(float(value), ('+' if signed else '') + f'.{digits}f').replace('-', '−')


def interval(value, limits, digits=3, signed=False):
    point = number(value, digits, signed)
    if limits is None or any(v is None for v in limits):
        return point
    return f'{point} [{number(limits[0], digits)}, {number(limits[1], digits)}]'


def coefficient(stage, model, feature):
    rows = stage['models'][model]['coefficients']
    return next(row for row in rows if row['feature'] == feature)


def coefficient_cell(row, digits=3):
    return interval(row['coefficient'], [row['ci95_low'], row['ci95_high']], digits, True)


def comparison(stage, reduced, full):
    return next((row for row in stage['paired_model_comparisons']
                 if row['reduced'] == reduced and row['full'] == full), None)


def family_direction(row):
    lo, hi = row.get('family40_ci_low'), row.get('family40_ci_high')
    if lo is None or hi is None:
        return 'Omitted'
    if lo > 0:
        return 'Positive'
    if hi < 0:
        return 'Negative'
    return 'Includes zero'


def unique_svg(svg, prefix):
    """Prevent duplicate matplotlib IDs when several SVGs share an HTML page."""
    svg = svg[svg.index('<svg'):]
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for old in sorted(set(ids), key=len, reverse=True):
        new = prefix + '-' + old
        svg = svg.replace('id="' + old + '"', 'id="' + new + '"')
        svg = svg.replace('#' + old + '"', '#' + new + '"')
        svg = svg.replace('#' + old + ')', '#' + new + ')')
    return svg


class Document:
    def __init__(self, destination, title):
        self.destination = destination
        self.title = title
        self.md, self.web = [], []

    def heading(self, title, level=2):
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        self.md.append('#' * level + ' ' + title + '\n')
        self.web.append(f'<h{level} id="{slug}">{html.escape(title)}</h{level}>')

    def paragraph(self, text, class_name=''):
        self.md.append(text + '\n')
        self.web.append(f'<p class="{class_name}">{html.escape(text)}</p>')

    def bullets(self, values):
        self.md.append('\n'.join('- ' + value for value in values) + '\n')
        self.web.append('<ul>' + ''.join('<li>' + html.escape(v) + '</li>' for v in values) + '</ul>')

    def table(self, headings, rows):
        def mdcell(value):
            return str(value).replace('|', '\\|').replace('\n', ' ')
        lines = ['| ' + ' | '.join(map(mdcell, headings)) + ' |',
                 '| ' + ' | '.join(['---'] + ['---:'] * (len(headings) - 1)) + ' |']
        lines += ['| ' + ' | '.join(map(mdcell, row)) + ' |' for row in rows]
        self.md.append('\n'.join(lines) + '\n')
        head = '<thead><tr>' + ''.join('<th scope="col">' + html.escape(str(h)) + '</th>' for h in headings) + '</tr></thead>'
        body = '<tbody>' + ''.join('<tr><th scope="row">' + html.escape(str(row[0])) + '</th>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in row[1:]) + '</tr>' for row in rows) + '</tbody>'
        self.web.append('<div class="table-scroll" tabindex="0"><table>' + head + body + '</table></div>')

    @contextmanager
    def details(self, title):
        self.md.append('<details>\n<summary>' + html.escape(title) + '</summary>\n')
        self.web.append('<details><summary>' + html.escape(title) + '</summary>')
        yield
        self.md.append('</details>\n')
        self.web.append('</details>')

    def links(self, links):
        self.md.append(' · '.join(f'[{label}]({path})' for label, path in links) + '\n')
        self.web.append('<p class="downloads">' + ' · '.join('<a href="' + html.escape(path, quote=True) + '">' + html.escape(label) + '</a>' for label, path in links) + '</p>')

    def figure(self, basename, description):
        svg = (self.destination / (basename + '.svg')).read_text()
        self.md.append(f'![{description}]({basename}.svg)\n')
        self.web.append('<figure role="group" aria-label="' + html.escape(description, quote=True) + '">' + unique_svg(svg, basename) + '<figcaption>' + html.escape(description) + '</figcaption></figure>')
        self.links([('SVG', basename + '.svg'), ('PNG', basename + '.png')])

    def save(self):
        (self.destination / 'RESULTS.md').write_text('\n'.join(self.md))
        page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>' + html.escape(self.title) + '</title><style>' + CSS + '</style></head><body><main>' + '\n'.join(self.web) + '</main></body></html>'
        (self.destination / 'report.html').write_text(page)


CSS = """
:root{color-scheme:light;--text:#222b30;--muted:#5f6970;--rule:#dce1e4;--accent:#286754}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--text);font-family:Georgia,'Times New Roman',serif;font-size:18px;line-height:1.6}
main{max-width:1140px;padding:54px 38px 90px;margin:auto}h1,h2,h3,summary,table,.downloads,.eyebrow,.notice{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
h1{font-size:clamp(29px,4vw,43px);line-height:1.16;letter-spacing:-.025em;font-weight:650;margin:8px 0 18px}h2{font-size:25px;line-height:1.3;margin:48px 0 17px;font-weight:650}h3{font-size:20px;margin:28px 0 10px}
p,li{max-width:88ch}p{margin:13px 0}ul{padding-left:24px}li{margin:8px 0}a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}
.eyebrow{font-size:14px;color:var(--muted);font-weight:550;letter-spacing:.015em}.notice{font-size:15px;border-left:4px solid #b28615;padding:12px 18px;background:#faf8ef}.downloads{font-size:14px;color:var(--muted)}
.table-scroll{overflow-x:auto;margin:20px 0 24px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid #eceff1}th:first-child{text-align:left}thead th{font-size:12px;letter-spacing:.015em;color:var(--muted);background:#f8f9f9;font-weight:650}tbody th{font-weight:550}tbody tr:last-child>*{border-bottom:0}
figure{margin:28px 0 8px}figure svg{display:block;width:100%;height:auto}figcaption{font-size:14px;line-height:1.5;color:var(--muted);max-width:100ch;margin:8px 0}
details{margin:22px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:12px 0}details+details{border-top:0}summary{cursor:pointer;font-size:16px;font-weight:550}details p{font-size:16px}details[open] summary{margin-bottom:18px}
@media(max-width:680px){body{font-size:17px}main{padding:28px 18px 60px}h2{font-size:23px;margin-top:38px}table{font-size:12px}th,td{padding:9px 10px}}
@media print{main{max-width:none;padding:0}h2{break-after:avoid}figure{break-inside:avoid}summary{color:#000}a{color:#000}.table-scroll{overflow:visible}table{font-size:10px}th,td{padding:6px}details{break-inside:avoid}}
"""


def plot_panels(data, destination, kind, preview=False):
    os.environ.setdefault('MPLCONFIGDIR', str(destination / '.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 10, 'axes.titlesize': 14,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#aab1b5',
                         'xtick.color': '#4f5960', 'ytick.color': '#303b42',
                         'svg.fonttype': 'none', 'svg.hashsalt': kind,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    spec = {
        'coefficients': ('resource-coefficients', 'Five-resource approximation', 'Coefficient (logit units per production pip)',
                         'Each stage is fitted separately. Lines show individual 95% board-cluster intervals.'),
        'enrichment': ('matched-resource-enrichment', 'Resource mix among matched alternatives', 'Chosen share / matched-baseline share',
                       'Same total pips and adjacent-tile count; pools of at least three corners. Lines show paired 95% intervals.'),
        'scarcity': ('board-scarcity-slopes', 'Association with board-wide resource scarcity', 'Logit / (local pip × board pip shortage)',
                     'Colored lines: individual 95% intervals. Thin gray lines: simultaneous family-of-40 Bonferroni intervals.'),
    }[kind]
    basename, title, xlabel, caption = spec
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1), sharex=True)
    extents = []
    for ax, stage in zip(axes.flat, data['stages']):
        for j, resource in enumerate(RESOURCES):
            outer = None
            if kind == 'coefficients':
                row = coefficient(stage, 'resource_pips', resource + '_pips')
                value, limits = row['coefficient'], [row['ci95_low'], row['ci95_high']]
            elif kind == 'scarcity':
                row = coefficient(stage, 'board_scarcity', resource + '_board_scarcity_interaction')
                value, limits = row['coefficient'], [row['ci95_low'], row['ci95_high']]
                outer = [row.get('family40_ci_low'), row.get('family40_ci_high')]
            else:
                stats = stage['enrichment']['matched_baseline']['statistics']
                value = stats.get('ratio_of_means', [None] * 5)[j]
                limits = stats.get('ratio_ci95', [None] * 5)[j]
            if value is None:
                ax.text(.02, j, 'Not estimable', transform=ax.get_yaxis_transform(), va='center', fontsize=9, color='#777')
                continue
            if outer and all(v is not None for v in outer):
                ax.hlines(j, *outer, colors='#a5adb2', linewidth=1.2, zorder=2)
                extents += outer
            if limits and all(v is not None for v in limits):
                ax.hlines(j, *limits, colors=COLORS[resource], linewidth=3, zorder=3)
                ax.plot(limits, [j, j], '|', color=COLORS[resource], ms=7, zorder=3)
                extents += limits
            extents.append(value)
            ax.plot(value, j, 'o', color=COLORS[resource], ms=6, zorder=4)
        ax.set_yticks(range(5), [r.title() for r in RESOURCES])
        ax.set_ylim(4.6, -.65)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.grid(axis='x', color='#e9ecee', linewidth=.7)
        ax.set_axisbelow(True)
        subtitle = stage['label']
        if kind == 'enrichment':
            eligible = stage['enrichment']['matched_baseline']['eligible_boards']
            subtitle += f'  ·  {eligible:,} matched boards'
        ax.set_title(subtitle, loc='left', pad=13, fontweight='semibold')
        if kind != 'coefficients':
            reference = 1 if kind == 'enrichment' else 0
            ax.axvline(reference, color='#78848b', linewidth=1, linestyle=(0, (4, 3)))
            extents.append(reference)
    if extents:
        lo, hi = min(extents), max(extents)
        span = max(hi - lo, .01)
        axes.flat[0].set_xlim(lo - span * .08, hi + span * .08)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, labelpad=12)
    if preview:
        title = 'Pilot preview — ' + title
    fig.suptitle(title, x=.09, ha='left', y=.985, fontsize=18, fontweight='semibold')
    fig.text(.09, .025, caption, ha='left', fontsize=9, color='#59646c')
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.14, hspace=.40, wspace=.32)
    fig.savefig(destination / (basename + '.svg'), metadata={'Date': None})
    fig.savefig(destination / (basename + '.png'), dpi=180, metadata={'Description': caption})
    plt.close(fig)
    return basename


def source_links(destination, data, data_root=None):
    """Copy small source snapshots; never package the checkpoint or datasets."""
    links = [('Full analysis JSON', 'results.json')]
    for label, filename in [('Coefficient CSV', 'coefficients.csv'), ('Model-fit CSV', 'models.csv'), ('Enrichment CSV', 'enrichment.csv')]:
        if (destination / filename).exists():
            links.append((label, filename))
    archive = (data_root or destination.parent) / 'source'
    copies = [
        ('Analysis source', 'opening_study_scale_analysis.py', data.get('analysis_source_sha256')),
        ('Board-generation source', 'opening_study_scale.py', None),
        ('Original feature definitions', 'opening_network_study.py', None),
        ('Study protocol', 'experiments/opening-scale-100k-protocol.md', None),
        ('Report-generation source', Path(__file__).name, None),
    ]
    provenance = []
    for label, filename, expected in copies:
        archived = archive / filename
        path = archived if archived.exists() else ROOT / filename
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verified = expected is None or digest == expected
        if not verified:
            # A pilot may precede a later source fix. Never label this the exact source.
            label += ' (current version; differs from recorded hash)'
        relative = Path('methods') / filename
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        links.append((label, relative.as_posix()))
        provenance.append(dict(label=label, original_path=str(path), sha256=digest, expected_sha256=expected, matches_recorded_hash=verified))
    for label, filename in [('Generation manifest', 'manifest.json'), ('Original-pipeline parity checks', 'original-study-parity.json'), ('Independent legality and inference validation', 'validation.json')]:
        path = (data_root or destination.parent) / filename
        if path.exists():
            relative = Path('methods') / filename
            shutil.copy2(path, destination / relative)
            links.append((label, relative.as_posix()))
    (destination / 'report-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    links.append(('Report source provenance', 'report-provenance.json'))
    return links


def render(source, formatting_test=False, data_root=None):
    destination = Path(source)
    result_path = destination / 'results.json'
    data = json.loads(result_path.read_text())
    assert data['schema'] == 'opening-scale-analysis-v1'
    assert data['resources'] == RESOURCES
    assert [s['label'] for s in data['stages']] == STAGES
    boards = int(data['boards'])
    preview = formatting_test or boards < 100_000
    for stage in data['stages']:
        assert stage['train_boards'] + stage['test_boards'] == boards
    figures = {kind: plot_panels(data, destination, kind, preview) for kind in ('coefficients', 'enrichment', 'scarcity')}
    downloads = source_links(destination, data, data_root)
    title = 'Generation 37 opening preferences'
    if preview:
        title = 'Formatting preview: opening-preference report'
    doc = Document(destination, title)
    doc.heading(title, 1)
    ntrain, ntest = data['stages'][0]['train_boards'], data['stages'][0]['test_boards']
    doc.paragraph(f'{boards:,} boards · {ntrain:,} fitting / {ntest:,} held out · {4 * boards:,} settlement decisions · A1 → B1 → B2 → A2', 'eyebrow')
    if preview:
        doc.paragraph('Formatting check only. This preview uses a small pilot to test layout and data handling; it is not the 100,000-board study and should not be used for scientific conclusions.', 'notice')
    doc.paragraph('Generation 37, the final generation in this training run, chooses one opening sequence per generated board using its highest-scoring legal move. The analyses below describe those network preferences. They use no search or completed-game outcomes, and do not estimate the winning value of a resource.')
    doc.links([('Markdown report', 'RESULTS.md'), ('Analysis data', 'results.json'), ('Methods and downloads', '#methods-and-downloads')])
    if not preview:
        doc.heading('The measured patterns')
        pip_values = [s['models']['total_pips']['r2'] for s in data['stages']]
        lines = [f'Total pips alone give held-out R² of {number(pip_values[0])} / {number(pip_values[1])} for the first settlements and {number(pip_values[2])} / {number(pip_values[3])} for the second settlements.']
        maxima = []
        for stage in data['stages']:
            ratios = stage['enrichment']['matched_baseline']['statistics'].get('ratio_of_means', [])
            candidates = [(v, r) for v, r in zip(ratios, RESOURCES) if v is not None]
            if candidates:
                value, resource = max(candidates)
                maxima.append(f'{stage["label"]}: {resource} ({value:.2f}×)')
        if maxima:
            lines.append('Largest chosen-share enrichment within the matched pools: ' + '; '.join(maxima) + '. These are descriptive ratios, not a universal resource ranking.')
        gains = [comparison(s, 'resource_context', 'board_scarcity')['delta_r2'] for s in data['stages']]
        lines.append(f'Adding board-supply interactions changes held-out R² by {number(min(gains), 4, True)} to {number(max(gains), 4, True)}, depending on placement stage. The detailed intervals and the separate legal-availability model appear below.')
        doc.bullets(lines)

    doc.heading('How well do simple features reproduce the scores?')
    doc.paragraph('A row of data is a legal settlement candidate. Scores and predictors are centered within each decision. R² = 1 − fitted mean squared error / equal-score mean squared error, with errors averaged first over candidates and then equally over boards. It measures relative-score fidelity, not the fraction of moves predicted correctly.')
    doc.table(['Regression inputs'] + STAGES, [[label] + [number(s['models'][key]['r2']) for s in data['stages']] for key, label in MODEL_LABELS.items()])
    doc.paragraph('The rows from five resource pip counts through resource-specific context add features cumulatively. The last two rows each add a different availability measure to that same context model; they are alternatives. Dice-number features include production variance, a nonlinear function of total production, so their gain cannot be assigned solely to diversification.')
    with doc.details('Choice agreement and held-out uncertainty'):
        doc.table(['Stage', 'Model', 'R² [95% CI]', 'Exact top-choice agreement'], [[s['label'], MODEL_LABELS[k], interval(s['models'][k]['r2'], s['models'][k]['r2_ci95']), f'{100 * s["models"][k]["top_choice_agreement"]:.1f}%'] for s in data['stages'] for k in ('resource_pips', 'full_context', 'resource_context')])
        doc.paragraph('A model can reproduce the broad score differences while missing close choices between the best corners. Test intervals use whole-board delta-method uncertainty conditional on the fitted regressions.')

    doc.heading('Each resource alone, then all five together')
    doc.paragraph('Each cell in the first table fits only that resource’s local pip count. These one-feature R² values do not add up to the joint fit and do not isolate resource importance.')
    doc.table(['Placement'] + [r.title() for r in RESOURCES], [[s['label']] + [number(s['models'][r + '_alone']['r2']) for r in RESOURCES] for s in data['stages']])
    doc.paragraph('The joint model fits five pip coefficients at once. The following values are logit units per pip in that approximation, with individual 95% intervals. They are not the neural network’s stored weights. Compare resources within one stage and specification; later stages also have different score scales and opening histories.')
    doc.figure(figures['coefficients'], 'Five-resource approximation coefficients, fitted separately at each opening stage. Horizontal lines show individual 95% board-cluster intervals.')
    doc.table(['Placement'] + [r.title() for r in RESOURCES], [[s['label']] + [coefficient_cell(coefficient(s, 'resource_pips', r + '_pips')) for r in RESOURCES] for s in data['stages']])
    with doc.details('Pairwise resource substitutions and multiplicity'):
        doc.paragraph('The following contrast replaces one pip of one resource with one pip of another inside the fitted five-resource equation. It does not assert that a valid board intervention could hold all other strategic properties fixed. Simultaneous intervals use Bonferroni adjustment over the 40 stage-specific pairwise contrasts.')
        doc.table(['Stage', 'Replace → add', 'Logit difference [95% CI]', 'Family-of-40 interval'], [[c['stage'], c['resource_replaced'].title() + ' → ' + c['resource_added'].title(), interval(c['logit_change_per_pip'], c['ci95'], signed=True), '[' + ', '.join(number(v) for v in c['family40_ci']) + ']'] for c in data.get('resource_substitution_contrasts', [])])

    doc.heading('What is selected when production and geometry match?')
    doc.paragraph('For each chosen settlement, compare its resource fractions with a uniformly selected legal corner having the same total pips and adjacent-tile count. The pool includes the chosen corner; retain a board only when at least three corners match. This conditions on the production level the network chose and covers the reported subset, rather than every opening.')
    doc.paragraph('Enrichment is the mean chosen resource fraction divided by the mean matched-baseline fraction, giving every eligible board equal weight. A ratio of 1.10 means a 10% larger average share of that resource’s pips; it does not mean the bot is 10% more likely to win or to select any particular corner.')
    coverage_rows = []
    for s in data['stages']:
        e = s['enrichment']['matched_baseline']; q = e.get('matching_pool_size_quantiles', [])
        pool = f'{number(q[2], 0)} ({number(q[1], 0)}–{number(q[3], 0)})' if len(q) == 5 else 'Not recorded'
        coverage_rows.append([s['label'], f'{e["eligible_boards"]:,} / {e["total_boards"]:,}', f'{100*e["eligible_boards"]/e["total_boards"]:.1f}%', pool])
    doc.table(['Placement', 'Eligible / all boards', 'Coverage', 'Matching pool: median (IQR)'], coverage_rows)
    doc.figure(figures['enrichment'], 'Chosen resource shares relative to matched legal alternatives. The dashed line marks no enrichment; intervals pair observations within the same board.')
    rows = []
    for s in data['stages']:
        stats = s['enrichment']['matched_baseline']['statistics']
        rows.append([s['label']] + [interval(v, ci, 3) for v, ci in zip(stats.get('ratio_of_means', [None]*5), stats.get('ratio_ci95', [None]*5))])
    doc.table(['Placement'] + [r.title() + ' ratio [95% CI]' for r in RESOURCES], rows)
    with doc.details('Already-owned versus missing resources, within matched pools'):
        rows = []
        for s in data['stages'][2:]:
            by_access = s['enrichment']['matched_baseline']['by_existing_access']
            for r in RESOURCES:
                missing, owned = by_access[r]['missing'], by_access[r]['already_owned']
                rows.append([s['label'], r.title(), f'{missing["boards"]:,}', interval(missing.get('enrichment'), missing.get('ci95')), f'{owned["boards"]:,}', interval(owned.get('enrichment'), owned.get('ci95'))])
        doc.table(['Stage', 'Resource', 'Missing: boards', 'Missing: ratio [CI]', 'Owned: boards', 'Owned: ratio [CI]'], rows)
        doc.paragraph('These subgroups condition on the first settlement chosen by the same network. They describe conditional preferences and do not hold the rest of the position fixed.')
    with doc.details('Comparison against all positive-production legal corners'):
        rows = []
        for s in data['stages']:
            stats = s['enrichment']['baseline']['statistics']
            rows.append([s['label']] + [interval(v, ci) for v, ci in zip(stats['ratio_of_means'], stats['ratio_ci95'])])
        doc.table(['Placement'] + [r.title() for r in RESOURCES], rows)
        doc.paragraph('This broader baseline includes differences in total production and coastal geometry. Zero-pip corners are excluded because their resource fractions are undefined. These descriptive calculations use all generated boards, including the fitting set; they do not fit another predictive model.')

    doc.heading('Does scarcity change the fitted resource preference?')
    doc.paragraph('The context model controls for the listed production, coverage, concentration, port, number-pattern, expansion, and proximity features. It also includes resource-specific interactions with existing production, missing access, matching owned ports, geometric port access, and starting-card counts. Two separate extensions then test board supply or the resources remaining among legal corners.')
    doc.bullets([
        'Board supply: candidate resource pips × (reference board supply − actual board supply). The reference is 58 × that resource’s tile count / 18; there are 58 pips across the 18 producing tiles.',
        'Legal availability: candidate resource pips × (reference board supply / 9 − mean resource pips among legal corners). At A1 this is exactly the board-supply interaction divided by nine, because every corner is legal.',
        'A positive interaction coefficient associates lower availability with a steeper per-pip score slope, conditional on this fitted specification. Board and legal-availability coefficients have different units and should not be compared directly.',
    ])
    doc.figure(figures['scarcity'], 'Board-supply interaction slopes after the specified context controls. Gray outer intervals adjust for 40 scarcity comparisons; positive slopes associate lower supply with greater per-pip priority in the approximation.')
    gain_rows = []
    for s in data['stages']:
        board = comparison(s, 'resource_context', 'board_scarcity')
        legal = comparison(s, 'resource_context', 'legal_scarcity')
        gain_rows.append([s['label'], number(s['models']['resource_context']['r2']), interval(board['delta_r2'], board['ci95'], 5, True), interval(legal['delta_r2'], legal['ci95'], 5, True)])
    doc.table(['Stage', 'Context R²', 'Add board supply: ΔR² [95% CI]', 'Add legal availability: ΔR² [95% CI]'], gain_rows)
    doc.paragraph('The gain is measured on the same held-out boards and its interval uses paired errors. A small coefficient interval can coexist with a small predictive gain. The A1 availability measures are algebraically equivalent, rather than two independent confirmations.')
    with doc.details('Scarcity coefficients, both specifications'):
        rows = []
        for s in data['stages']:
            for r in RESOURCES:
                board = coefficient(s, 'board_scarcity', r + '_board_scarcity_interaction')
                legal = coefficient(s, 'legal_scarcity', r + '_legal_scarcity_interaction')
                rows.append([s['label'], r.title(), coefficient_cell(board, 4), family_direction(board), coefficient_cell(legal, 4), family_direction(legal)])
        doc.table(['Stage', 'Resource', 'Board coefficient [95% CI]', 'Adjusted sign', 'Legal coefficient [95% CI]', 'Adjusted sign'], rows)
        doc.paragraph('Adjusted sign indicates whether the Bonferroni family-of-40 interval excludes zero. These are conditional associations, not causal responses to changing a board. A1 appears in both specifications but uses the same availability information.')

    doc.heading('Availability and context counts')
    doc.paragraph('Wood, sheep, and wheat occupy four tiles each; brick and ore occupy three. Scarcity here concerns their production pips or legal accessibility, not a changing count of resource tiles.')
    totals = data['stages'][0]['availability']['mean_board_pips']
    doc.table(['Resource', 'Tiles', 'Mean pips on the full board'], [[r.title(), count, number(value)] for r, count, value in zip(RESOURCES, [4, 3, 4, 4, 3], totals)])
    doc.table(['Mean pips per legal corner'] + [r.title() for r in RESOURCES], [[s['label']] + [number(v) for v in s['availability']['mean_legal_candidate_pips']] for s in data['stages']])
    with doc.details('Support for conditional resource and port terms'):
        rows = []
        for s in data['stages']:
            for r in RESOURCES:
                support = s.get('context_support', {}).get(r, {})
                fallback = s['enrichment']['baseline']['by_existing_access'][r]
                missing = support.get('missing_resource_boards', fallback['missing']['boards'])
                own = support.get('already_owned_resource_boards', support.get('owned_resource_boards', fallback['already_owned']['boards']))
                port = support.get('owned_matching_port_boards')
                rows.append([s['label'], r.title(), f'{missing:,}', f'{own:,}', f'{port:,}' if port is not None else 'Not recorded'])
        doc.table(['Stage', 'Resource', 'Missing production', 'Existing production', 'Owned matching 2:1 port'], rows)
        doc.paragraph('Counts refer to all generated boards at that stage, not only matched pools. Rare port contexts may have much less support than the overall sample size suggests. Constant and redundant columns are omitted, and an omitted coefficient does not mean that the network ignores that feature.')

    doc.heading('Methods and downloads')
    doc.paragraph('Every stage has its own regression. The 80/20 split keeps all four positions from a board together. Features are scaled using training-only within-decision RMS values; ridge regularization is fixed at 0.001. Coefficient intervals use a board-cluster sandwich that includes uncertainty in those RMS scales. Held-out intervals condition on the fitted regressions. Enrichment intervals use paired, board-level log-ratio uncertainty.')
    doc.paragraph('Individual 95% intervals are descriptive intervals for this specified analysis. The two labeled families of 40 contrasts use Bonferroni adjustment; other tables are not simultaneous tests. All intervals condition on the trained checkpoint and do not measure variation across independently trained agents.')
    doc.bullets([
        'The larger sample reduces board-sampling noise. It does not remove confounding, approximation error, or biases of the frozen policy.',
        'B1, B2, and A2 arise from earlier greedy choices. The results do not cover all legal opening histories or arbitrary human play.',
        'The matched comparison fixes only total pips and adjacent-tile count. Ports, particular resources, neighboring opportunities, and opponent positions can still differ.',
        'No full games or internal interventions were run. The report measures neither optimal openings nor effects on winning probability.',
    ])
    doc.paragraph('Checkpoint SHA-256: ' + data['model']['sha256'], 'downloads')
    doc.paragraph('Analysis-source SHA-256 recorded with the results: ' + data['analysis_source_sha256'], 'downloads')
    doc.paragraph('Validation: candidate choices match the saved policy argmax; training estimating equations close numerically; A1 board/availability fits agree. Exact numerical checks are retained in the analysis JSON.')
    with doc.details('Exact regression feature sets and method metadata'):
        for key, label in MODEL_LABELS.items():
            doc.heading(label, 3)
            doc.paragraph(', '.join(data['model_definitions'][key]))
        for label, method in data.get('method', {}).items():
            doc.paragraph(label.replace('_', ' ').capitalize() + ': ' + method)
    doc.links(downloads)
    doc.save()
    print(json.dumps(dict(report=str(destination / 'report.html'), markdown=str(destination / 'RESULTS.md'), boards=boards, formatting_preview=preview, figures=list(figures.values())), indent=2))
    return destination / 'report.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT, help='Analysis directory containing results.json.')
    parser.add_argument('--data-root', type=Path, help='Optional generator run directory holding source snapshots.')
    parser.add_argument('--formatting-test', action='store_true', help='Clearly mark output as a pilot formatting preview.')
    args = parser.parse_args()
    render(args.source.resolve(), args.formatting_test, args.data_root.resolve() if args.data_root else None)


if __name__ == '__main__':
    main()
