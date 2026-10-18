import importlib.util

import numpy as np
import pandas as pd
import torch


def preparation():
    assert importlib.util.find_spec('integrations.physionet_baselines') is not None, 'PhysioNet pipeline is missing'
    from integrations.physionet_baselines import prepare
    return prepare


def test_quarter_hour_bins_keep_last_valid_measurement_and_exclude_future():
    p = preparation()
    rows = pd.DataFrame({'Time':['00:00','00:14','00:15','00:16','00:20','24:00','24:01','48:00','48:01'],
        'Parameter':['HR']*9,'Value':[1.,2.,3.,100.,-1.,4.,5.,6.,999.]})
    values, observed, ends = p.aggregate_record(rows, ['HR'])
    assert values.shape == (192,1)
    assert values[[0,1,95,96,191],0].tolist() == [3.,100.,4.,5.,6.]
    assert np.isnan(values[2,0]) and not observed[2,0]
    assert ends[[0,95,191]].tolist() == [900.,86400.,172800.]


def test_unknown_and_static_fields_are_not_dynamic_truth():
    p = preparation()
    rows = pd.DataFrame({'Time':['00:00']*5,'Parameter':['Age','RecordID','MechVent','HR','pH'],
        'Value':[99.,123456.,1.,-1.,np.inf]})
    values, observed, _ = p.aggregate_record(rows, ['HR','pH'])
    assert not observed.any() and np.isnan(values).all()


def test_record_split_is_reproducible_disjoint_and_has_seventy_ten_twenty_ratio():
    p = preparation()
    ids=[str(i) for i in range(100)]
    a=p.record_splits(ids,2023);b=p.record_splits(ids[::-1],2023)
    assert a==b and len(a)==100
    assert [list(a.values()).count(s) for s in ['train','validation','test']]==[70,10,20]
    assert set(a)==set(ids)


def test_fill_never_uses_another_record_or_future():
    p = preparation()
    values=np.array([[1.],[np.nan],[np.nan],[4.]])
    segments=[dict(start=0,end=2),dict(start=2,end=4)]
    assert p.fill_segments(values,segments,np.array([7.]))[:,0].tolist()==[1.,1.,7.,4.]


def test_scaler_ignores_placeholders_and_nontraining_records():
    p = preparation()
    values=np.array([[1.,2.],[999.,4.],[3.,999.],[999999.,999999.]])
    valid=np.array([[True,True],[False,True],[True,False],[True,True]])
    mean,std=p.observed_training_stats(values,valid,3)
    np.testing.assert_allclose(mean,[2.,3.]);np.testing.assert_allclose(std,[1.,1.])


def test_one_window_per_complete_patient_record_and_no_stitching():
    p = preparation()
    segments=[dict(start=0,end=192,split='train'),dict(start=192,end=384,split='validation'),
        dict(start=384,end=574,split='test')]
    starts=p.record_window_starts(segments)
    assert {s:v.tolist() for s,v in starts.items()}=={'train':[0],'validation':[192],'test':[]}


def test_empty_supervision_is_finite_zero_with_zero_gradient():
    preparation()
    from integrations.physionet_baselines.losses import mean_or_zero
    x=torch.tensor([4.,7.],requires_grad=True)
    loss=mean_or_zero(x[torch.tensor([False,False])].square())
    loss.backward()
    assert loss.item()==0. and x.grad.tolist()==[0.,0.]


def test_sparse_supervision_only_changes_valid_gradient():
    preparation()
    from integrations.physionet_baselines.losses import hidden_error, mean_or_zero
    x=torch.tensor([[[2.,1000.]]],requires_grad=True)
    truth=torch.zeros_like(x);seen=torch.zeros_like(x,dtype=torch.bool)
    loss=mean_or_zero(hidden_error(x,truth,seen,torch.tensor([[[True,False]]])).square())
    loss.backward()
    assert loss.item()==4. and x.grad.tolist()==[[[4.,0.]]]
