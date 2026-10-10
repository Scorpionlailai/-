from resource_summary import combine_stage_costs


def test_cpu_projection_does_not_erase_measured_gpu_peaks_or_training_cost():
    records=[dict(status='completed',full_workflow_seconds=7.,peak_allocated_bytes=None,
                  peak_reserved_bytes=None,device={'type':'cpu','name':'x86_64'}),
             dict(status='completed',full_workflow_seconds=13.,peak_allocated_bytes=2*2**30,
                  peak_reserved_bytes=3*2**30,device={'type':'cuda','name':'GPU'})]
    result=combine_stage_costs(records)
    assert result['full_training_seconds']==20.
    assert result['peak_allocated_gib']==2.
    assert result['peak_reserved_gib']==3.


def test_unknown_gpu_peak_remains_unknown():
    records=[dict(status='completed',full_workflow_seconds=13.,peak_allocated_bytes=None,
                  peak_reserved_bytes=None,device={'type':'cuda','name':'GPU'})]
    result=combine_stage_costs(records)
    assert result['peak_allocated_gib'] is None
    assert result['peak_reserved_gib'] is None
