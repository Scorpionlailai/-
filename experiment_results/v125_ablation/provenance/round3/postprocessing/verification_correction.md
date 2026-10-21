# 最终核验脚本行数修正

原封存verify3.py保留，原supervisor失败记录一并归档。配对表有3训练seed × 2划分 × 4阶段 × 2评价范围 × 2指标＝96行；原断言误写48。隔离副本仅将此断言改为96并逐一核验完整笛卡尔积、唯一键，无缺失和重复。其余原核验全部保留，修正后通过。没有重训、重评估或修改任何指标、掩码、checkpoint和训练协议。

```diff
--- original_sealed/verify3.py
+++ postprocessing/verify3_rowcount_corrected.py
@@ -86,7 +86,11 @@
         take=[y for y in seeds if all(x[k]==y[k] for k in ('variant','split','phase','section'))];assert len(take)==3
         for m in ('mse','mae'):
             values=[float(y[m]) for y in take];assert abs(float(x[m+'_mean'])-np.mean(values))<1e-13 and abs(float(x[m+'_training_std_ddof1'])-np.std(values,ddof=1))<1e-13
-    pairs=list(csv.DictReader(open(DEST/'round3_paired_differences.csv')));assert len(pairs)==48
+    pairs=list(csv.DictReader(open(DEST/'round3_paired_differences.csv')))
+    import itertools
+    expected_pairs=set(itertools.product((2023,2024,2025),('validation','test'),('coarse','calibration64','calibration256','ensemble'),('full96','recent32'),('mse','mae')))
+    actual_pairs=[(int(x['training_seed']),x['split'],x['phase'],x['section'],x['metric']) for x in pairs]
+    assert len(pairs)==len(expected_pairs)==96 and set(actual_pairs)==expected_pairs and len(set(actual_pairs))==len(actual_pairs)
     for x in pairs:assert abs(float(x['A4_minus_A0'])-(float(x['A4_error'])-float(x['A0_error'])))<1e-13
     # Baseline A0/A1 numeric observations remain identical after the append.
     published={(x['variant'],x['training_seed'],x['split'],x['mask_seed'],x['phase'],x['section']):x for x in mask}
```
