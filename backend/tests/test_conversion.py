"""Domain tests for the instrument-conversion and relay-chain handoff workflow."""

import numpy as np
import pytest

from src.domain.conversion import (
    analyze_conversion,
    merge_pair_rows,
    parse_conversion_settings,
)
from src.domain.instruments import (
    NIR_FEATURES,
    absorbance,
    fit_edges,
    normalize_pairs,
    strip_index,
)
from src.domain.project import analyze, demo_pairs, demo_rows
from src.domain.settings import SPEC


@pytest.fixture(scope='module')
def base_result():
    return analyze(demo_rows(), {'max_components': 6, 'screening_mpa': 4.2})


def _report(rows, base_result, settings=None, previous=None, anchor='NIR-01'):
    return analyze_conversion(rows, base_result, anchor, settings or {'max_hops': 2, 'error_mpa': 0.30},
                              4.2, previous)


def _handoff_block(report):
    return report['handoff']


def test_conversion_settings_bounds():
    assert parse_conversion_settings({}) == (2, 0.30)
    assert parse_conversion_settings({'max_hops': 3, 'error_mpa': 0.1}) == (3, 0.1)
    for bad in ({'max_hops': 0}, {'max_hops': 5}, {'max_hops': 1.5}, {'error_mpa': 0.0},
                {'error_mpa': 6}, {'unknown': 1}):
        with pytest.raises(ValueError):
            parse_conversion_settings(bad)


def test_normalize_pairs_rejects_duplicate_and_split_leak():
    base = {'split': 'train', 'instrument': 'NIR-01', 'id': 'X', 'batch_id': 'B',
            **{field: 0.3 for field in NIR_FEATURES}, 'density_kg_m3': 800.0,
            'thickness_mm': 12.0, 'flexural_mpa': 4.0}
    with pytest.raises(ValueError):
        normalize_pairs([base, dict(base)])
    with pytest.raises(ValueError):
        normalize_pairs([base, {**base, 'instrument': 'NIR-02', 'batch_id': 'B2'}])
    with pytest.raises(ValueError):
        normalize_pairs([{**base, 'split': 'pending', 'flexural_mpa': 4.0}])
    rows = normalize_pairs([base, {**base, 'instrument': 'NIR-02'}])
    assert len(rows) == 2


def test_edges_round_trip_anchor(base_result):
    records = normalize_pairs(demo_pairs())
    edges = fit_edges(records)
    index = strip_index(records)
    for strip in index.values():
        converted = edges[('NIR-02', 'NIR-01')].intercept + edges[('NIR-02', 'NIR-01')].slope * absorbance(strip['NIR-02'])
        assert np.allclose(converted, absorbance(strip['NIR-01']), atol=1e-3)


def test_clean_pairs_handoff_is_ready(base_result):
    report = _report(demo_pairs(), base_result)
    block = _handoff_block(report)
    assert block['ready'] is True
    assert block['reasons'] == []
    chain_table = next(table for table in report['tables'] if table['key'] == 'relay_chains')
    assert all(row['qualified'] for row in chain_table['rows'])
    assert len(block['rows']) == 30


def test_acceptance_uses_only_holdout_batches(base_result):
    """Chains must be scored on stripes that were not part of any edge fit."""

    report = _report(demo_pairs(), base_result)
    chain_table = next(table for table in report['tables'] if table['key'] == 'relay_chains')
    for row in chain_table['rows']:
        assert row['acceptance_batches'] == 3
        assert row['acceptance_strips'] == 18


def test_missing_chain_defers_handoff(base_result):
    # Anchor instrument has data, but replacement-instrument pending strips have no
    # train/holdout pairing, so no chain can be fitted for them.
    rows = [row for row in demo_pairs()
            if row['instrument'] == 'NIR-01' or (row['instrument'] == 'NIR-02' and row['split'] == 'pending')]
    report = _report(rows, base_result)
    block = _handoff_block(report)
    assert block['ready'] is False
    assert block['rows'] == []
    assert any('没有合格链' in reason for reason in block['reasons'])


def test_strict_error_line_defers_when_no_chain_qualifies(base_result):
    # Bend replacement-instrument holdout readings nonlinearly, so a linear edge
    # fitted on the train strips cannot reproduce the anchor estimate for any chain.
    rows = []
    for row in demo_pairs():
        row = dict(row)
        if row['instrument'] in ('NIR-02', 'NIR-03') and row['split'] == 'holdout':
            for field in NIR_FEATURES:
                row[field] = round(row[field] + 0.06 * row[field] ** 2, 6)
        rows.append(row)
    report = _report(rows, base_result, settings={'max_hops': 2, 'error_mpa': 0.05})
    block = _handoff_block(report)
    assert block['ready'] is False
    assert block['rows'] == []
    chain_table = next(table for table in report['tables'] if table['key'] == 'relay_chains')
    assert not any(row['qualified'] for row in chain_table['rows'])


def test_split_across_screening_line_is_disputed():
    """Directly exercise the dispute rule: qualified chains on opposite sides of the line."""

    from src.domain.conversion import _pending_analysis
    from src.domain.instruments import Chain

    class Model:
        def predict(self, x):
            return 4.0 + 2.0 * x[:, 0]

        def outside(self, x):
            return np.zeros((x.shape[0], len(NIR_FEATURES)), dtype=bool)

    record = {'id': 'X-1', 'batch_id': 'B', 'split': 'pending', 'instrument': 'NIR-02',
              **{field: 0.40 for field in NIR_FEATURES}, 'density_kg_m3': 800.0,
              'thickness_mm': 12.0, 'flexural_mpa': None}
    record['nir_1200'] = 0.10
    # Two qualified chains produce 4.18 (below) and 4.26 (above) respectively.
    below = Chain(1, ('NIR-02', 'NIR-01'), np.array([-0.01, 0, 0, 0, 0, 0]), np.ones(6))
    above = Chain(2, ('NIR-02', 'NIR-03', 'NIR-01'), np.array([0.03, 0, 0, 0, 0, 0]), np.ones(6))
    evaluated = [
        {'chain': below, 'qualified': True, 'rmse': 0.01, 'bias': 0.0, 'max_abs': 0.0, 'note': '',
         'acceptance_batches': 1, 'acceptance_strips': 2, 'acceptance_rows': []},
        {'chain': above, 'qualified': True, 'rmse': 0.02, 'bias': 0.0, 'max_abs': 0.0, 'note': '',
         'acceptance_batches': 1, 'acceptance_strips': 2, 'acceptance_rows': []},
    ]
    _, strips = _pending_analysis(Model(), [record], evaluated, 4.2)
    assert len(strips) == 1
    assert strips[0]['status'] == 'disputed'
    assert strips[0]['chain_sides'] == ['低于筛查线', '不低于筛查线']


def test_revision_keeps_only_supported_strips(base_result):
    original = _handoff_block(_report(demo_pairs(), base_result))
    previous = {'handoff_id': 11, 'rows': original['rows'], 'ready': original['ready']}
    # Degrade NIR-02 holdout only, so the direct and relay chains through NIR-02 fail.
    rows = []
    for row in demo_pairs():
        row = dict(row)
        if row['instrument'] == 'NIR-02' and row['split'] == 'holdout':
            for field in NIR_FEATURES:
                row[field] = round(row[field] + 0.05, 6)
        rows.append(row)
    report = _report(rows, base_result, previous=previous)
    block = _handoff_block(report)
    assert block['revision'] is True
    assert block['previous_handoff_id'] == 11
    assert block['ready'] is True
    instruments = {row['instrument'] for row in block['rows']}
    assert 'NIR-02' not in instruments
    assert all(item['id'] for item in block['dropped'])
    assert {item['status'] for item in block['dropped']} <= {'unconverted', 'judgement_changed', 'no_current_reading'}


def test_revision_does_not_add_new_strips(base_result):
    original = _handoff_block(_report(demo_pairs(), base_result))
    previous = {'handoff_id': 12, 'rows': original['rows'], 'ready': original['ready']}
    rows = list(demo_pairs())
    extra = dict(rows[0])
    extra.update(id='GP-P99-01', batch_id='GP-P99', split='pending', instrument='NIR-02', flexural_mpa=None)
    rows.append(extra)
    report = _report(rows, base_result, previous=previous)
    block = _handoff_block(report)
    assert all(row['id'] != 'GP-P99-01' for row in block['rows'])
    assert any(item['id'] == 'GP-P99-01' and item['status'] == 'not_in_original' for item in block['dropped'])


def test_merge_pair_rows_rejects_conflicting_reads():
    first = demo_pairs()
    second = [dict(row) for row in first]
    second[0]['nir_1200'] = round(second[0]['nir_1200'] + 0.5, 6)
    with pytest.raises(ValueError):
        merge_pair_rows([first, second])
    merged = merge_pair_rows([first, [dict(row) for row in first[:3]]])
    assert len(merged) == len(first)


def test_spec_exposes_paired_schema(base_result):
    assert 'instrument' in SPEC['paired_columns']
    keys = [item['key'] for item in SPEC['conversion_settings']]
    assert keys == ['max_hops', 'error_mpa']
