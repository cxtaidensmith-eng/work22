"""Assemble an offline, Chinese figure guide and export archive."""
import csv
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'Visualization/c_shared_mapping_20261003'
EXPERIMENT = ROOT/'Result/shared_mapping_seed0_20261003'
DIAG = ROOT/'Result/c_shared_mapping_visualization_20261003/relation_paths'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    summary = read(EXPERIMENT/'summary.json')
    audit = read(EXPERIMENT/'audit.json')
    diagnostic = read(DIAG/'COMPLETE.json')
    path_values = read(OUT/'fig05_metadata.json')
    union_file = DIAG/'posthoc_union_and_numerical_caution.json'
    union = read(union_file)['subsets']['all_36']
    (OUT/'fig05_union_and_numerical_caution.json').write_bytes(union_file.read_bytes())
    with (DIAG/'fold_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    near_zero_count = sum(abs(float(row['delta_nll'])) < 1e-6 for row in diagnostic_rows)
    assert audit['passed'] and audit['verified_checkpoints'] == 20
    assert summary['all_new_complete'] and diagnostic['passed']
    assert diagnostic['checkpoints'] == 10 and diagnostic['interventions'] == 360 and diagnostic['no_training']
    figures = []
    for i in range(1, 6):
        matches = sorted(OUT.glob(f'fig{i:02d}_*.png'))
        assert len(matches) == 1, (i, matches)
        for ext in ('pdf','svg'):
            assert matches[0].with_suffix('.'+ext).exists()
        figures.append(matches[0])
    c, shared = summary['variants']['C'], summary['variants']['SHARED_MAP']
    pair = summary['matched_comparisons'][0]
    best, last = pair['stages']['best'], pair['stages']['last']
    rows = []
    for label, left, right in (
            ('选定最佳点 ACC', c['best']['acc'], shared['best']['acc']),
            ('选定最佳点宏 F1', c['best']['macro_f1'], shared['best']['macro_f1']),
            ('选定最佳点 AUC', c['best']['auc'], shared['best']['auc']),
            ('第 1000 轮 ACC', c['last']['acc'], shared['last']['acc']),
            ('末 100 轮平均 ACC', c['mean_last100_acc'], shared['mean_last100_acc'])):
        rows.append(f'<tr><td>{label}</td><td>{left*100:.2f}%</td><td>{right*100:.2f}%</td><td>{100*(right-left):+.2f}</td></tr>')
    titles = ['逐折对比与训练阶段差异', '训练全过程', '患者纠错与类别差异',
              '发送／接收映射的参数分布', '关系来源 × 传播内容的依赖热图']
    descriptions = [
        f'使用相同十折划分配对比较。按各折首次最高测试 ACC 选点，共享版比原 C 低 {abs(best["mean_acc_delta"])*100:.2f} 个百分点；原 C 在 {best["fold_losses"]} 折更高、{best["fold_ties"]} 折相同、{best["fold_wins"]} 折更低。末轮与末 100 轮为预先报告的辅助指标，不能事后替代主指标。',
        '展示每一轮的十折测试准确率及两模型差异。图中若使用平滑，窗口与方向在图注中明确；折间标准差描述划分之间的离散程度，不是置信区间。各折最佳点的平均值不等于平均曲线的最高值。训练后期共享版表现更好，不能据本次结果宣称 C 在整个训练过程持续占优。',
        f'所有计数均为共享版相对原 C：最佳点纠正 {best["corrected"]} 人、新增错误 {best["harmed"]} 人，净 {best["net"]:+d} 人；末轮纠正 {last["corrected"]} 人、新增错误 {last["harmed"]} 人，净 {last["net"]:+d} 人。每名患者只使用其所在测试折的预测，患者计数与等权折均准确率分别报告。',
        '从原 C 与共享版的全部 10 个最佳检查点提取映射。发送与接收尺度以及相对旋转角用于描述传播算子的参数行为；相对角采用周期处理。尺度小不等于该模态不重要，尺度大也不代表更可靠。两次传播、度归一化、内容投影及后续融合共同决定最终作用。',
        '对原 C 的每个最佳检查点分别做 36 次推理干预：只屏蔽一个“关系来源 × 内容模态”通道中的跨患者分量，保留完整两次传播中的自身对角项（含二跳返回）、原边权、映射和 1/6 来源汇总。左图为干预后的负对数似然变化，右图为新增错误减去纠正错误；正值表示移除该路径有害。每个格子是独立干预，格间数值不可相加，不能解释为临床因果效应。',
    ]
    cards = []
    for i, (figure, title, caption) in enumerate(zip(figures, titles, descriptions), 1):
        links = ' · '.join(f'<a href="{html.escape(figure.with_suffix("."+ext).name)}">{ext.upper()}</a>' for ext in ('png','pdf','svg'))
        cards.append(f'<section id="fig{i}"><div class="figtop"><h2>图 {i} · {title}</h2><span>{links}</span></div><p>{caption}</p><a href="{html.escape(figure.name)}"><img src="{html.escape(figure.name)}" alt="{title}"></a></section>')
    names = ['MRI', 'PET', 'CSF', '风险因素', '认知量表', 'ROI 平均指标']
    matrix = path_values['mean_delta_nll']
    cross = sorted([(matrix[r][m], r, m) for r in range(6) for m in range(6) if r != m], reverse=True)
    top = cross[0]
    net_matrix = path_values['net_added_errors']
    net_harm_count = sum(value > 0 for row in net_matrix for value in row)
    net_improvement_count = sum(value < 0 for row in net_matrix for value in row)
    if top[0] > 0:
        value, r, m = top
        mechanism = (f'新增机制诊断：36 条路径分别移除后，{net_harm_count} 条使汇总正确人数净下降，'
                     f'{net_improvement_count} 条使其净增加；这些是路径数量，不是不同患者人数。'
                     f'全部干预真正改变预测的只有 {union["unique_changed_patients"]} 名不同患者。'
                     f'跨模态路径中，移除后平均 NLL 增加最大的是“{names[r]}关系 → {names[m]}内容”，'
                     f'但幅度仅 {value:+.5f}。当前结果不能证明单条跨模态路径不可替代。'
                     '接近零的正负变化不作机制解释，全部数值均为固定检查点上的探索诊断。')
    else:
        mechanism = '本次所有跨模态单路径移除的平均 NLL 均未增加，不能据热图主张这些单路径提供了稳定的正收益。'
    for name in ('fold_metrics.csv','aggregate.csv'):
        (OUT/('fig05_'+name)).write_bytes((DIAG/name).read_bytes())
    csv_links = ''.join(f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in sorted(OUT.glob('*.csv')))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>C 框架 · 共享映射消融与机制可视化</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#253449;font-family:"Microsoft YaHei","Noto Sans SC",Arial,sans-serif;line-height:1.75}main{max-width:1220px;margin:auto;padding:38px 24px 60px}h1{font-size:29px;line-height:1.4;margin:10px 0}h2{font-size:20px;margin:0}.eyebrow{letter-spacing:2px;font-size:12px;font-weight:bold;color:#226b9c}.intro{max-width:1020px}section,.summary{background:white;border:1px solid #dde4ec;border-radius:12px;margin:24px 0;padding:24px}img{width:100%;height:auto;display:block;margin-top:15px}a{color:#1b6392;text-decoration:none}a:hover{text-decoration:underline}.figtop{display:flex;justify-content:space-between;gap:20px;align-items:baseline}.figtop span{white-space:nowrap;font-size:13px}p{margin:10px 0}.note{color:#596d82;font-size:14px}table{border-collapse:collapse;width:100%;font-size:15px;margin-top:16px}td,th{padding:11px 13px;border-bottom:1px solid #e4eaf0;text-align:right}td:first-child,th:first-child{text-align:left}th{color:#596d82;font-weight:normal}nav{display:flex;flex-wrap:wrap;gap:16px;margin-top:22px}.tag{background:#eaf2f8;border-radius:20px;padding:4px 12px;color:#226b9c;font-size:13px}li{overflow-wrap:anywhere}@media(max-width:650px){main{padding:20px 12px}.figtop{display:block}section,.summary{padding:16px}table{font-size:12px}td,th{padding:8px 4px}h1{font-size:24px}}
    </style><main><div class="eyebrow">TAD · MODEL C · VISUAL DIAGNOSTICS</div><h1>C 框架：共享映射消融与机制可视化</h1><p class="intro">这一组图同时回答两个问题：按模态区分发送／接收规则带来了多少分类收益，以及原 C 如何依赖不同的关系来源和传播内容。全部使用已完成训练的日志与检查点，本次不新增训练。</p><nav><span class="tag">seed0 · 固定十折</span><span class="tag">每折 1000 轮</span><span class="tag">10 个 C 检查点 · 360 次路径干预</span><span class="tag">PNG / PDF / SVG + CSV</span></nav>
    '''
    document += '<div class="summary"><h2>主要发现</h2><p><strong>模态专属映射提高了原选轮协议下的最佳表现，但优势有限；共享映射的训练后期表现更好。</strong></p><table><thead><tr><th>指标</th><th>原 C</th><th>共享映射</th><th>共享 − C（百分点）</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    document += f'<p>{mechanism}</p><p class="note">最佳点沿用首次最高测试 ACC 的选择规则；本轮为单种子探索分析。共享映射版保留相同映射生成器及全部传播路径，活跃参数仅减少 80 个，但初始映射也随共享而变化。原 C 主线与历史实验结果未修改。</p></div>'
    document += ''.join(cards)
    document += f'<p class="note">数值边界：路径诊断有 {near_zero_count}/{len(diagnostic_rows)} 个逐折 ΔNLL 的绝对值小于 10⁻⁶；这些极小变化的符号不作为可靠效应证据。某一路径移除无损并不等于整套传播无效，路径之间可能有冗余或非线性交互，尚需独立检验。</p>'
    document += '<section><h2>可复核数据</h2><p>矢量图适合论文排版；CSV 保留绘图数值。实验日志、检查点和源文件校验信息保存在本地研究目录，本图包不包含患者标识。</p><ul>'+csv_links+'</ul><p><a href="visualization_manifest.json">图包来源与文件校验清单</a></p></section></main></html>'
    (OUT/'index.html').write_text(document, encoding='utf-8')
    files = [p for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'visualization_manifest.json']
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        no_new_training=True, fixed_checkpoint_interventions=360, checkpoints=10,
        shared_ablation_audit=dict(verified_runs=audit['verified_runs'], verified_checkpoints=audit['verified_checkpoints'], passed=audit['passed']),
        figures=len(figures), metadata=[p.name for p in files if p.suffix=='.json'],
        source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (EXPERIMENT/'summary.json',EXPERIMENT/'audit.json',DIAG/'COMPLETE.json',DIAG/'aggregate.csv',union_file,Path(__file__))},
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        interpretation='Exploratory test-selected fixed checkpoints; descriptive variability, no inferential or clinical causal claims.')
    (OUT/'visualization_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive = OUT.parent/'c_shared_mapping_visualization_20261003.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file(): bundle.write(path,arcname=f'c_shared_mapping_20261003/{path.name}')
    print(json.dumps(dict(report=str(OUT/'index.html'),archive=str(archive),files=len(files)+1,mechanism=mechanism),ensure_ascii=False))


if __name__ == '__main__': main()
