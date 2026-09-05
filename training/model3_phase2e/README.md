# Model 3 Phase 2E.1

This research-only area reconstructs the Challenge2015 portion of Model 3
using the nested `training.zip` in `mitbih-vtvtf.zip`. It does not train Model
3, export ONNX, modify `vfvt_feature_table.csv`, or change production code.

## Calibration

The MAT `val` arrays were inspected as signed digital samples. For each
record, the selected channel's `.hea` line supplies `adc_zero`, `gain`, and
`baseline`. The experimental physical-unit conversion is:

```text
physical_mV = (digital_value - adc_zero) / gain + baseline
```

This is justified by the MAT first samples matching the header ADC-zero values
and the header declaring channel-II units as `mV`. The notebook's universal
`CHALLENGE2015_GAIN = 200` is never used.

## Selection and labels

- Input sampling rate: 250 Hz
- Output sampling rate: 360 Hz via `resample_poly`
- Selected channel: `II` only
- Labels: true `Ventricular_Tachycardia -> VT` and
  `Ventricular_Flutter_Fib -> VF` alarms only
- Other rhythms and false alarms are excluded
- Records without channel II are reported and excluded
- AFIB is explicitly rejected; the archive scan found no AFIB text

The same ten feature definitions are regenerated through the existing feature
function, without changing that production implementation. No Model 3
training is performed in this phase.

Run:

```powershell
python -m training.model3_phase2e.model3_phase2e_reconstruct vfvt_feature_table.csv mitbih-vtvtf.zip --output-dir training/model3_phase2e/results
```

Outputs are research reports only. The reconstruction covers Challenge2015;
full cross-source reconstruction still requires raw VFDB and CUDB archives.