from pathlib import Path
import os

OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'svg.fonttype': 'none',
                     'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(12.8, 6.7), dpi=220)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 1280), ylim=(670, 0))
ax.set_aspect('equal')
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

INK = '#203040'
MUTED = '#677684'
TEAL = '#137f86'
LINE = '#7d8a96'

def text(x, y, s, size=12, color=INK, weight='normal', ha='center', **kwargs):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va='center', linespacing=1.5, **kwargs)

def box(x, y, w, h, face='#f4f6f8', edge='#b9c3cc', linewidth=1.1, radius=12):
    patch = FancyBboxPatch((x,y), w,h,
        boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=face,
        edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    return patch

def line(points, color=LINE, width=1.5):
    ax.plot([p[0] for p in points], [p[1] for p in points],
            color=color, linewidth=width, solid_capstyle='round', zorder=1)

def arrow(start, end, color=LINE, width=1.5, scale=13):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
        mutation_scale=scale, linewidth=width, color=color,
        shrinkA=0, shrinkB=0, zorder=2))

text(640, 39, '框架 C：关系从哪里来，内容如何传播？', 22, weight='bold')
text(640, 78, '先保留模态身份进行患者间传播，再进行模态融合', 12.5, MUTED)

# Input and the two independent axes supplied to the propagation module.
box(30, 170, 170, 105)
text(115, 202, '多模态输入', 15, weight='bold')
text(115, 241, 'MRI / PET / …', 12, MUTED)

box(245, 120, 255, 92, '#f2f7fa', '#b7cbd8')
text(372.5, 145, '模态关系图', 14, weight='bold')
text(372.5, 172, '用什么找邻居？', 11.5, MUTED)
text(372.5, 196, 'MRI 图 / PET 图 / …', 10.5, MUTED)

box(245, 245, 255, 92, '#f5f3fa', '#c7bfd8')
text(372.5, 270, '模态内容编码', 14, weight='bold')
text(372.5, 297, '从邻居传什么？', 11.5, MUTED)
text(372.5, 321, 'MRI 表示 / PET 表示 / …', 10.5, MUTED)

line([(200, 222), (222, 222), (222, 166)])
arrow((222,166), (245,166))
line([(222,222), (222,291)])
arrow((222,291), (245,291))

# Only the two contribution-bearing design choices receive visual emphasis.
box(580, 110, 385, 260, '#ecf8f7', TEAL, 1.8, 16)
text(772.5, 146, '① 关系来源与内容分开', 15, TEAL, 'bold')
text(772.5, 181, '每种内容利用多种关系图', 12)
line([(605,207), (940,207)], '#c2dfdd', 1)
text(772.5, 242, '② 按内容调整传播方式', 15, TEAL, 'bold')
text(772.5, 280, '边权结合关系先验与内容相似性', 11)
text(772.5, 307, '发送／接收变换按内容模态区分', 11)
text(772.5, 345, '分别传播 → 关系平均 → 残差更新', 10.5, MUTED)
arrow((500,166), (580,166), TEAL, 1.8)
arrow((500,291), (580,291), TEAL, 1.8)

box(1040, 210, 210, 100)
text(1145, 241, '模态融合与类别读取', 12, weight='bold')
text(1145, 279, '双分支相加 → 分类', 11, MUTED)
arrow((965,248), (1040,248), TEAL, 1.8)

# The original patient-only branch is present but visually subordinate.
line([(115,275), (115,409), (1145,409), (1145,331)], '#bfab96', 1.2)
arrow((1145,331), (1145,310), '#bfab96', 1.2)
text(640,409,'自身信息分支',10.5,'#8f7963',
     bbox=dict(facecolor='white', edgecolor='none', pad=6))

# One concrete example makes the two axes understandable without graph jargon.
box(30, 462, 1220, 165, '#faf9fc', '#dedbe5', 1, 14)
text(58, 488, '例：用 MRI 关系，补充 PET 内容', 14, '#675887', 'bold', ha='left')
box(160, 537, 275, 62, '#f1ecfa', '#c8bcdc')
box(845, 537, 275, 62, '#f1ecfa', '#c8bcdc')
text(297.5,568,'邻居患者 j 的 PET 表示',12,'#57446f')
text(982.5,568,'目标患者 i 的 PET 表示',12,'#57446f')
arrow((455,568),(825,568),'#8d78b0',2,15)
text(640,541,'MRI 决定候选邻居',10.5,'#675887')
text(640,596,'采用 PET 模态的传播变换',10.5,'#675887')
text(640,651,'核心机制示意：省略门控、网络层数和训练损失。',9,MUTED)

for suffix in ('png','svg'):
    path = OUT / f'framework_c_innovation_explainer.{suffix}'
    fig.savefig(path, dpi=220, facecolor='white', metadata={'Title':'框架 C 的核心设计'})
    print(path)
plt.close(fig)
