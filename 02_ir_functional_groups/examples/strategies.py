"""One explicit list shared by preparation, training and comparison runners."""
STRATEGIES = [
    dict(key='01a', name='01a omit + deduplicate', notebook='01a_IR_Omit_Transmittance.ipynb',
         directory='examples/outputs/01a_omit_transmittance', signal='absorbance_only',
         bounds='require_full_coverage', resampling='linear', cleansing='canonical_deduplication'),
    dict(key='01b', name='01b convert + label check', notebook='01b_IR_Convert_Transmittance.ipynb',
         directory='examples/outputs/01b_convert_transmittance', signal='convert_transmittance',
         bounds='require_full_coverage', resampling='linear', cleansing='smarts_label_check'),
    dict(key='01c', name='01c pad + quality filter', notebook='01c_IR_Padding_and_Quality.ipynb',
         directory='examples/outputs/01c_padding_and_quality', signal='convert_transmittance',
         bounds='zero_pad', resampling='linear', cleansing='roughness_quantile'),
    dict(key='01d', name='01d bin average + balance', notebook='01d_IR_Resampling_and_Balancing.ipynb',
         directory='examples/outputs/01d_resampling_and_balancing', signal='convert_transmittance',
         bounds='zero_pad', resampling='bin_average_then_upsample', cleansing='capped_rare_oversampling'),
]
