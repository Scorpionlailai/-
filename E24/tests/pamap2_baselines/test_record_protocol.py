"""Catch stitching through dropped rows, future aggregation and cross-record fill."""
import importlib.util

import numpy as np


def preparation():
    assert importlib.util.find_spec('integrations.pamap2_baselines') is not None, 'PAMAP2 pipeline is missing'
    from integrations.pamap2_baselines import prepare
    return prepare


def test_segments_keep_removed_transitions_and_clock_gaps_as_boundaries():
    p = preparation()
    timestamps = np.array([.01,.02,.03,.04,.05,.06,1.,1.01,1.02,1.03])
    labels = np.array([1,1,0,1,1,2,2,2,0,2])
    assert p.segment_runs(timestamps, labels) == [(0,2,1),(3,5,1),(5,6,2),(6,8,2),(9,10,2)]


def test_10hz_bins_use_only_past_observations():
    p = preparation()
    timestamps = np.array([.01,.04,.10,.11,.20,.21])
    values = np.array([[1.,np.nan],[np.nan,np.nan],[5.,np.nan],[10.,3.],[20.,np.nan],[9999.,8.]])
    result, observed, ends = p.aggregate_10hz(timestamps,values)
    assert ends.tolist() == [.1,.2,.3]
    assert result[:2,0].tolist() == [3.,15.]
    assert np.isnan(result[0,1]) and result[1,1] == 3.
    assert observed.tolist() == [[True,False],[True,True],[True,True]]


def test_forward_fill_does_not_borrow_from_previous_record_or_future():
    p = preparation()
    values = np.array([[1.],[np.nan],[np.nan],[4.]])
    segments = [dict(start=0,end=2),dict(start=2,end=4)]
    filled = p.fill_segments(values,segments,np.array([7.]))
    assert filled[:,0].tolist() == [1.,1.,7.,4.]


def test_windows_require_both_history_and_target_in_one_record():
    p = preparation()
    segments = [dict(start=0,end=220,split='train'),
                dict(start=220,end=412,split='validation'),
                dict(start=412,end=600,split='test')]
    starts = p.record_window_starts(segments)
    assert {s:x.tolist() for s,x in starts.items()} == {'train':[0,16],'validation':[220],'test':[]}


def test_subjects_are_disjoint_across_all_sessions():
    p = preparation()
    assert [p.subject_split(s) for s in range(101,110)] == ['train']*6+['validation','test','test']
