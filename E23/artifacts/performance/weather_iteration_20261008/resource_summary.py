"""Resource accounting for a measured workflow containing CPU and GPU stages."""


def combine_stage_costs(records):
    if not records or any(record['status'] != 'completed' for record in records):
        raise ValueError('Only completed resource stages can be aggregated')
    gpu = [record for record in records if record['device']['type'] == 'cuda']
    result = {'full_training_seconds': sum(record['full_workflow_seconds'] for record in records)}
    for source, key in [('peak_allocated_bytes','peak_allocated_gib'),
                        ('peak_reserved_bytes','peak_reserved_gib')]:
        values = [record[source] for record in gpu]
        result[key] = max(values)/2**30 if values and all(value is not None for value in values) else None
    return result
