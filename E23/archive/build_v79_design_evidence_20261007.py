"""Build a design explanation, calculated gains and a checksummed evidence ZIP."""
from pathlib import Path
import csv,hashlib,json,os,re,zipfile
from datetime import datetime,timezone

PAPER=Path('/home/xyw/paper');ROOT=PAPER/'E23';DOCS=ROOT/'docs/experiments'
EXPS=ROOT/'artifacts/ETTh1/seq96_pred96/experiments'
V79=EXPS/'e23_v79_v69_global_level_four_pureMSE_20261007'
BASE=EXPS/'e23_t1_pure_mse_frozen_four_scenarios_20261007'
OUT=ROOT/'artifacts/archives/E23_v79_design_evidence_20261007'
OUT.mkdir(parents=True,exist_ok=False)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def readcsv(p):
 with Path(p).open() as f:return list(csv.DictReader(f))

def writecsv(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

history=readcsv(DOCS/'ETTh1_E23_v69_signature_position_sr015_bl8_20261005.csv')
by={r['model']:r for r in history}
pairs=[
 ('可见CNN前端','E23-v41-slow','E23-v46-front-context','新增window CNN；历史单seed开发对照'),
 ('冻结前端','E23-v46-front-context','E23-v47-frozen-front','复用同一Stage1；更新政策控制'),
 ('点级专家路由','E23-v47-frozen-front','E23-v48-point-router','由patch仲裁细化到点级'),
 ('稳健局部坐标','E23-v51-candidate-readout','E23-v54-local-coordinates','同一候选读出基础的坐标修改'),
 ('整窗signature注意力','E23-v58-recent-mean','E23-v63-plain-signature-MAE','主线阶段对照；不是严格单模块重训消融'),
 ('全部冻结修复器','E23-v63-plain-signature-MAE','E23-v64-retained-frozen-MAE','复用同一Stage1；Stage2更新政策控制'),
 ('变量×时间位置参数','E23-v65-seed2023-frozen-MAE','E23-v69-seed2023-frozen-MAE','同seed/冻结政策；历史混合预测损失'),
]
module_rows=[]
for module,parent,child,scope in pairs:
 a,b=by[parent],by[child]
 module_rows.append(dict(module=module,parent=parent,child=child,scene='sr015_bl8',parent_seed=a['training_seed'],child_seed=b['training_seed'],parent_mse=float(a['mse']),child_mse=float(b['mse']),mse_reduction=float(a['mse'])-float(b['mse']),mse_reduction_percent=100*(float(a['mse'])-float(b['mse']))/float(a['mse']),parent_mae=float(a['mae']),child_mae=float(b['mae']),mae_reduction=float(a['mae'])-float(b['mae']),mae_reduction_percent=100*(float(a['mae'])-float(b['mae']))/float(a['mae']),evidence_scope=scope,source=str(DOCS/'ETTh1_E23_v69_signature_position_sr015_bl8_20261005.csv')))
module_csv=DOCS/'E23_v79_key_module_historical_improvements_20261007.csv';writecsv(module_csv,module_rows)
main=readcsv(DOCS/'ETTh1_E23_v79_v69_T1_pureMSE_frozen_four_scenarios_20261007.csv')
main=[r for r in main if r['selection_metric']=='mae']
lookup={(r['scene'],r['model']):r for r in main}
scenes=['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16'];labels={'sr010_bl8':'0.10/8','sr015_bl8':'0.15/8','sr020_bl8':'0.20/8','sr010_bl16':'0.10/16'}
gain_rows=[]
for scene in scenes:
 own=lookup[scene,'E23-v79-pureMSE']
 for parent in ['E23-v69-pureMSE','T1-pureMSE-frozen']:
  ref=lookup[scene,parent]
  gain_rows.append(dict(scene=scene,reference=parent,model='E23-v79',split='test',mask_bank='development20',selection_metric='mae',reference_mse=float(ref['mse_mean']),model_mse=float(own['mse_mean']),mse_reduction=float(ref['mse_mean'])-float(own['mse_mean']),mse_reduction_percent=100*(float(ref['mse_mean'])-float(own['mse_mean']))/float(ref['mse_mean']),reference_mae=float(ref['mae_mean']),model_mae=float(own['mae_mean']),mae_reduction=float(ref['mae_mean'])-float(own['mae_mean']),mae_reduction_percent=100*(float(ref['mae_mean'])-float(own['mae_mean']))/float(ref['mae_mean'])))
gain_csv=DOCS/'E23_v79_matched_forecast_improvements_20261007.csv';writecsv(gain_csv,gain_rows)
repair_rows=[]
for scene in scenes:
 stage2=json.loads((BASE/'E23-v69'/scene/'stage2/runs/e23-stage2.json').read_text())
 parent=json.loads(Path(stage2['stage1_manifest']).read_text())['validation_repair_metrics']
 own=json.loads((V79/scene/'E23-v79/stage1/runs/e23-stage1.json').read_text())['validation_repair_metrics']
 repair_rows.append(dict(scene=scene,split='fixed_validation',parent='E23-v69',model='E23-v79',parent_mse=parent['global_mse'],model_mse=own['global_mse'],mse_reduction=parent['global_mse']-own['global_mse'],parent_mae=parent['global_mae'],model_mae=own['global_mae'],mae_reduction=parent['global_mae']-own['global_mae']))
repair_csv=DOCS/'E23_v79_v69_stage1_repair_comparison_20261007.csv';writecsv(repair_csv,repair_rows)
handoff=(PAPER/'E24/docs/2026-10-07_E23_to_E24_handoff.md').read_text()
architecture=handoff.split('## 3. E24 起点：v79 的具体构成\n',1)[1].split('\n## 4.',1)[0]
text='''# E23-v79：设计过程、关键模块与实测改进

日期：2026-10-07。本文只整理已有记录，没有训练、重新选模或新增测试；后台E24 Weather不受影响。v79为封存基础模型，本轮材料整理不启动E23迭代队列。

## 1. 先抓住核心设计

我们要解决的是：历史窗口缺失时，如何提取足够可信的历史信息，交给保持不变的预测器。v79先在修复器内形成表示和候选，再显式恢复96点历史，最终由原PatchTST预测未来96点。

它有三条相互配合的设计线：

1. **证据从哪里来**：当前时间T、当前其他变量V、训练历史原型H；单轮条件读取T→V→H。
2. **证据怎样到达每个缺失点**：可见CNN和整窗上下文、八头候选读出、点级路由、重叠置信融合、末32点精修。
3. **恢复的数值尺度怎样可靠**：全局修复MSE训练/选模、稳健局部坐标、近期均值与跨变量全局水平校正。

Stage2完全冻结修复器，把“修复学习”与“预测器学习”分开；这既防止修复表示被预测梯度继续改写，也提供与冻结T1匹配的下游比较。

```mermaid
flowchart TD
    A[96点可见历史与原始mask] --> B[可见CNN与整窗signature上下文]
    B --> T[T：当前变量时间证据]
    T --> V[V：随窗口选择其他变量来源]
    V --> H[H：查询训练历史原型]
    H --> C[八头点级候选读出与none/H/V/T路由]
    C --> F[重叠置信融合与末32点精修]
    A --> G[7变量真实可见统计]
    G --> L[近期均值与全局水平校正]
    F --> L
    L --> R[显式完整历史：观测点原值不变]
    R --> P[原PatchTST预测未来96点]
```

图示为职责关系；路由的专家名称仍是H/V/T，不意味着有八套完整三专家。模型内的顺序更新也不构成因果识别。

## 2. 设计为何逐步走到这里

### 2.1 最初三专家：先回答信息来源

H补充训练历史形状，V利用同窗变量联系，T利用同变量时间连续性。早期width64的v3固定验证推理消融：全模型MSE0.764770；关闭H/V/T分别变为0.826515/0.831872/0.792099。三类证据在该已训练权重中都有效，V的关闭损失最大。这是v3推理消融，不是v79重训练消融，不能直接当作v79三个专家的贡献排名。

### 2.2 宽度16→64：增加修复表示容量，保持预测器不变

同一早期P4固定测试，宽度16→64使预测MSE0.448187→0.429694、MAE0.439265→0.437349。之后只改Stage2选模政策为验证MAE，v3测试为0.427265/0.434449。前者是容量变化，后者是选模变化；两种收益不能都归给模型结构。它们只有单seed/单组位置，是早期探索结果。

### 2.3 V从全局固定来源变为当前窗口来源

v12保留4个名额，依据当前窗口/patch证据选来源。验证MSE0.763652→0.745448、MAE0.589049→0.588212（相对v11）；其单组测试0.425613/0.431006。作用是让可用来源随缺失位置变化，而非在所有窗口使用相同四变量。H的分组重排v13没有进一步改善，故未继承。

### 2.4 v28解决评分尺度失真

旧v12中，可见std<0.01的缺失坐标只占0.672678%，却贡献窗口归一化平方误差98.991745%；换到训练全局标准化尺度后仅贡献0.286937%。v28改为缺失坐标 `((修复值−真值)×query_std)^2`，按同尺度验证MSE选Stage1，保留0.1检索损失。

同尺度修复MSE从v12的0.368824变为v28的0.305816，降17.08%。本轮训练内部按全局指标选第99轮，按窗口指标则选第26轮，确实出现不同选模。这里同时改了损失形状和尺度，不能拆分两者的独立贡献。

### 2.5 单轮条件读取优于盲目增加重复

v29先用T/V形成当前证据，再由更新后的表示查询H。单轮的三位置预测0.410121/0.423863，三轮0.424405/0.439457。三轮Stage1修复可降至0.255366，但最终预测没有更好；v30大堆叠Stage1修复0.205550也未转成预测优势。最终保留单轮。这不是“重复一定无用”，而是现有重复形式未找到收益。

### 2.6 高缺失下，从patch级仲裁走向可见上下文和点级取舍

v46增加可见CNN；v48增加点级路由；v51增加八头候选读出；v54在同一读出基础上改稳健局部坐标。重要的是：v51单独在当时配方中预测退步，v54坐标处理才使这套读出组合恢复收益。因此八头模块目前有功能解释和组合证据，尚没有证明它在v79中单独一定增益的移除重训练消融。

重叠patch仍用P16/S8，通过confidence融合处理同一点多个输出；最近32点精修补足末端粒度。v49/v50的开发测试也没有每项都同时改善MSE/MAE，不能把所有子模块写成独立单调增益。

### 2.7 近期水平、整窗关系、位置与冻结

近期均值项让模型关注末32点的level；v63去掉此前失败的大统计调制，保留整窗signature注意力；v69加变量×时间位置参数。Stage2全冻结控制显示，保住Stage1信息比继续小步修改修复器更好。以下给出可追溯的历史配对数字。

## 3. 当前v79完整模块和训练配置

'''+architecture+'''

## 4. 关键模块的历史收益：有数据，但不夸大证据

以下全部为sr015_bl8、旧20组测试均值，按各版固定验证MAE选权重。除位置参数配对为seed2023，其余为seed502。它们是不同历史实验配方的局部配对，不是把整个v79逐模块拆除后重训练的完整消融。正的下降百分比代表改善。

| 设计变化 | 父版→新版 | MSE父→新 | MSE下降 | MAE父→新 | MAE下降 |
|---|---|---:|---:|---:|---:|
'''
for r in module_rows:
 text+=f'| {r["module"]} | {r["parent"]} → {r["child"]} | {r["parent_mse"]:.6f} → {r["child_mse"]:.6f} | {r["mse_reduction_percent"]:.2f}% | {r["parent_mae"]:.6f} → {r["child_mae"]:.6f} | {r["mae_reduction_percent"]:.2f}% |\n'
text+='''
- **证据较直接**：全部冻结控制复用了相同Stage1；v54是v51基础上的坐标改动；v69位置参数有相同seed/冻结政策的参考。
- **阶段性证据**：v58→v63包含中间父版/训练路径，不能把全部差值归为signature单模块的因果贡献。
- **尚需补充**：v79本身的CNN、H/V/T、八头读出、置信融合、末32点精修等逐模块移除重训练；不同seed/数据集重现。不能据当前历史轨迹宣称每个模块在所有场景都必要。

## 5. v79相对v69到底新增了什么

只将 `recent_mean_projection: mean_v1` 改为 `global_level_mean_v2`：在原均值校正器旁增加 `Linear(49,64)→GELU→Linear(64,1)`，共3265参数。它不改预测器。

每个变量取6项真实可见统计：整96点均值、末32点均值、最近可见值、整窗可见比例、末32点可见比例、可见query_std。7变量共42项，拼接目标变量7维one-hot，共49项。新分支不读取隐藏真值，也不把粗修复值当真实统计；整窗都无观测时输出新修正0。

新分支输出 `δ=2*tanh(g/2)`，单位是训练全局标准化值；与原均值支路的修正相加，只施加到末32点缺失坐标。再乘 `share=min(32/末32点缺失数,4)` 并除query_std写回局部接口；可见点硬保留。**±2是新分支乘share前的界，不是整个最终修复值或总修正的界。** 最后一层零初始化，构造时隔离RNG，保持父版初始函数与公共头初始化。

设计动机：粗候选在局部相对坐标中可能形状合理，却有近期整体level偏移。其他变量的当前真实可见水平为level提供额外参照。这个解释与模块输入/作用方式一致，但需消融验证，不能仅由最终误差推断因果机制。

### 同配方v69→v79：四场景预测

双方纯MSE、全冻结、原头、seed2023、100轮；旧20组均值。每格MSE/MAE，下降为v79相对v69。

| 起点率/块长 | v69 | v79 | MSE下降 | MAE下降 |
|---|---:|---:|---:|---:|
'''
for scene in scenes:
 a=lookup[scene,'E23-v69-pureMSE'];b=lookup[scene,'E23-v79-pureMSE'];r=next(x for x in gain_rows if x['scene']==scene and x['reference']=='E23-v69-pureMSE')
 text+=f'| {labels[scene]} | {float(a["mse_mean"]):.6f} / {float(a["mae_mean"]):.6f} | {float(b["mse_mean"]):.6f} / {float(b["mae_mean"]):.6f} | {r["mse_reduction_percent"]:.2f}% | {r["mae_reduction_percent"]:.2f}% |\n'
text+='''
### 同时检查Stage1修复，避免只挑预测收益

| 起点率/块长 | v69全窗验证修复MSE/MAE | v79全窗验证修复MSE/MAE |
|---|---:|---:|
'''
for r in repair_rows:text+=f'| {labels[r["scene"]]} | {r["parent_mse"]:.6f} / {r["parent_mae"]:.6f} | {r["model_mse"]:.6f} / {r["model_mae"]:.6f} |\n'
text+='''
v79的预测四场景两指标都改善，但修复MSE只有0.10/8改善，其余三个略退步。因此不能说global-level分支“全面提高修复质量”，更不能只用预测提升证明最近32点已修得更好。

## 6. 与T1的当前完整结论

'''+(DOCS/'ETTh1_E23_v79_T1_forecast_full_repair_last32_pause_20261007.md').read_text().split('## 全窗口未来96点预测：',1)[1].join(['## 全窗口未来96点预测：',''])+'''

## 7. 实验口径与当前证据边界

- ETTh1 96→96、7变量；训练/验证/测试边界0/8640/11520/14400，训练段全局标准化。四个场景0.10/8、0.15/8、0.20/8、0.10/16，允许块重叠；起点率不是最终点缺失率。
- 当前比较双方公共Stage2均纯MSE、全冻结、原头同初始SHA、batch128/FP32、Adam lr1e-4、100轮/6600更新；MAE主选，MSE次选同轨迹另存，不混入主表。
- v79 Stage1 seed2023、100轮/6600更新；T1 Stage1 seed502、作者batch16/BF16/早停预算。不是Stage1优化器预算完全相同的比较，差异披露。
- 旧20组mask seed2026100401..20；新确认20组2026100701..20；两套都已公开。单权重20组位置的std不是训练seed的std；v79没有完整五seed结论。
- 全窗/末32点修复是固定验证诊断，尚缺正式20组测试修复。v79目前并未在预测、全窗修复、末32点修复三项上四场景全面击败T1。
- v79两阶段成本约839–871秒、allocated约4.51–4.67GiB；T1侧含原Stage1缓存成本与新Stage2实测，资源范围见原CSV。完整训练时间包括准备、全部轮次、验证/保存，排除最终测试。
- 近期/变量关系是统计证据，不构成已识别因果关系。oracle替换诊断只读取固定验证真值，不进入训练；不能把oracle预测当模型成绩或理论上界。

## 8. 对外表述与后续补证

目前可以表述：以时间、变量、训练历史三类证据为基础，结合点级仲裁、稳健数值恢复和近期跨变量level校正，形成可接原预测器的缺失历史修复模块；在锁定的纯MSE冻结下游配方中，v79相对匹配v69改善四场景预测均值，且在新位置确认中对T1有三个场景双指标优势。

仍需补证：v79逐模块重训练消融、全/末32点20组测试修复、多训练seed、跨领域数据集。八头、重叠融合、近期精修的职责可以解释，独立贡献应以这些消融判断。E24 Weather正在提供跨领域实验，本材料不读取其运行中结果。

## 9. 证据文件

- `E23_v79_key_module_historical_improvements_20261007.csv`：历史局部配对收益与范围。
- `E23_v79_matched_forecast_improvements_20261007.csv`：v79相对匹配v69/T1的四场景绝对/相对提升。
- `E23_v79_v69_stage1_repair_comparison_20261007.csv`：新增分支前后的修复变化。
- ZIP中的 `FILE_INDEX.csv` 给出每个文件的原始路径、包内位置、大小和SHA256；`PACKAGE_README.md`说明范围和旧路径查验方式。原始记录均保留字节内容，没有为可移植性重写历史manifest。
'''
design=DOCS/'2026-10-07_E23_v79_design_and_improvement_evidence.md'
with design.open('x') as f:f.write(text)
print(json.dumps(dict(event='documents_ready',document=str(design),csvs=[str(module_csv),str(gain_csv),str(repair_csv)]),ensure_ascii=False),flush=True)
# Include the core formal comparisons, exact selected weights and negative controls.
files=set()
def add(p):
 p=Path(p)
 if p.is_file() and not p.is_symlink():files.add(p)
 elif p.is_dir() and not p.is_symlink():
  for folder,dirs,names in os.walk(p,followlinks=False):
   dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache') and not (Path(folder)/d).is_symlink()]
   for name in names:
    q=Path(folder)/name
    if q.suffix not in ('.pyc','.pyo') and not q.is_symlink():files.add(q)
for p in [design,module_csv,gain_csv,repair_csv]:add(p)
patterns=['E23_width64*','E23_v11_v12*','E23_v28*','E23_v29*','E23_v30*','E23_v31*','2026-10-05_E23_evolution*','2026-10-06_E23_full*','ETTh1_E23_v69*','ETTh1_E23_v79*','ETTh1_E23_v80*','ETTh1_E23_v81*','ETTh1_E23_v82*','2026-10-05_E23_v46*','2026-10-05_E23_v48*','2026-10-05_E23_v51*','2026-10-05_E23_v54*','2026-10-05_E23_v58*','2026-10-05_E23_v63*']
for pattern in patterns:
 for p in DOCS.glob(pattern):add(p)
for p in [V79,BASE,EXPS/'e23_four_scene_confirmation_reserved_20261007',EXPS/'e23_v79_toward_clean_20261007']:add(p)
# Historical supporting raw records, without retaining every historical binary checkpoint.
for row in history:
 for field in ['selection','evaluation']:
  val=row.get(field,'')
  if val and Path(val).is_file():
   p=Path(val);add(p)
   record=json.loads(p.read_text())
   for key in ['stage1_manifest','repair_selection_manifest']:
    if record.get(key):add(record[key])
   if field=='selection':
    for rp in (p.parent.parent/'resources').glob('*.json'):add(rp)
# Recursively retain saved parents and exact weights for the selected matched comparison.
queue=list(p for p in files if p.suffix=='.json' and (p.is_relative_to(V79) or p.is_relative_to(BASE)))
seen=set()
while queue:
 p=queue.pop()
 if p in seen:continue
 seen.add(p)
 try:doc=json.loads(p.read_text())
 except (ValueError,OSError):continue
 def walk(value):
  if isinstance(value,dict):
   for key,v in value.items():
    if isinstance(v,str) and key in ('checkpoint','stage1_manifest','repair_selection_manifest','t1_repair_selection','test_bank','scenario_manifest','protocol_config','e23_config','mask_artifacts'):
     target=Path(v)
     if target.is_file() and target.is_relative_to(PAPER):
      add(target)
      if target.suffix=='.json':queue.append(target)
      if key=='test_bank':add(target.parent)
     elif key=='mask_artifacts' and target.is_dir():
      for q in target.glob('**/*'):
       if q.is_file() and ('mask' in q.name or q.name.endswith('manifest.json')):add(q)
    walk(v)
  elif isinstance(value,list):
   for v in value:walk(v)
 walk(doc)
for scene in scenes:
 q=BASE/'T1'/scene/'training/forecast-mae-selection.json';record=json.loads(q.read_text())
 for field in ['repair_selection_manifest']:
  if record.get(field):
   p=Path(record[field]);add(p);parent=json.loads(p.read_text());add(parent['checkpoint'])
   for resource in (p.parent/'resources').glob('*.json'):add(resource)
add(PAPER/'data/raw/ETTh1.csv')
add(PAPER/'E22/migration/paper_catalog.jsonl')
for p in [ROOT/'archive/E23-v79_seal_manifest_20261007.json',ROOT/'archive/legacy_path_map_20261007.json',ROOT/'archive/resolve_legacy_path.py',PAPER/'E24/docs/2026-10-07_E23_to_E24_handoff.md',PAPER/'E24/docs/EXPERIMENT_PROTOCOL.md',PAPER/'E24/configs/experiment_protocol_v1.json']:add(p)
# Exact model source is already inside v79/frozen-source.tar.gz; include the actual active config as a convenience copy.
add(ROOT/'versions/e23-v79-global-level-four/E23/configs/e23_v79_v69_global_level_seed2023.yaml')
index=[dict(original_path=str(p),archive_path='E23_v79_design_evidence_20261007/paper/'+str(p.relative_to(PAPER)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
writecsv(OUT/'FILE_INDEX.csv',index)
(OUT/'PACKAGE_README.md').write_text('''# E23-v79 设计与实验依据包\n\n先读 `paper/E23/docs/experiments/2026-10-07_E23_v79_design_and_improvement_evidence.md`。\n\n内容：设计脉络与三份提升统计CSV、相关历史报告及支持JSON、四场景v79/v69/T1匹配纯MSE冻结记录、主/次选权重、确认20位置结果与mask、Clean/修复诊断、源码冻结归档、配置/资源/CSV数据与可校验文件索引。保留v80-v82负结果报告。\n\n没有复制E24运行中的Weather文件；没有启动训练。历史文件保持原SHA和绝对路径，不是脱离环境的一键运行镜像。`FILE_INDEX.csv`用于定位包内文件；旧源码目录移动由`paper/E23/archive/legacy_path_map_20261007.json`说明。完整旧实验仍在原workspace。v79源码位于正式实验的`frozen-source.tar.gz`。作者基线源码/运行环境未整仓重复打包，revision和source SHA留在原记录。\n\n历史阶段配对与当前统一配方分开，不将不同seed、缺失参数、联合/冻结政策或单组/20组数字混为模块消融。修复为固定验证诊断，正式20组测试修复尚未补齐。\n''')
zip_path=OUT.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for row in index:z.write(row['original_path'],row['archive_path'])
 for p in [OUT/'FILE_INDEX.csv',OUT/'PACKAGE_README.md']:z.write(p,'E23_v79_design_evidence_20261007/'+p.name)
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 for row in index:
  with z.open(row['archive_path']) as f:
   h=hashlib.sha256()
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  assert h.hexdigest()==row['sha256'],row['archive_path']
result=dict(status='verified',created_utc=datetime.now(timezone.utc).isoformat(),document=str(design),archive=str(zip_path),archive_bytes=zip_path.stat().st_size,archive_sha256=sha(zip_path),indexed_files=len(index),zip_crc_verified=True,every_file_sha256_verified=True,training_performed=False,weather_running_outputs_included=False)
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
