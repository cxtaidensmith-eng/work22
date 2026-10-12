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
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(12, 15.4), dpi=180)
fig.patch.set_facecolor('white')
ax.set(xlim=(0,1200), ylim=(1540,0), aspect='equal')
ax.axis('off')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)

def txt(x,y,s,size=12,bold=False,color='#303030',**kwargs):
    ax.text(x,y,s,ha='center',va='center',fontsize=size,
            fontweight='bold' if bold else 'normal',color=color,
            linespacing=1.5,**kwargs)

def box(x,y,w,h,core=False,face='#f5f5f5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle='round,pad=0,rounding_size=16',
        facecolor=face,edgecolor='#262626' if core else '#c3c3c3',
        linewidth=2.1 if core else 1.1,zorder=2))

def line(points,color='#929292',width=1.4):
    ax.plot([p[0] for p in points],[p[1] for p in points],
        color=color,lw=width,solid_capstyle='round',zorder=1)

def arrow(a,b,color='#737373',width=1.4):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
        linewidth=width,color=color,shrinkA=0,shrinkB=0,zorder=1))

txt(600,48,'框架 C：关系与内容分开，按内容传播',22,True,'#181818')
txt(600,88,'先保留模态表示进行群体传播，再完成融合与诊断',12,color='#707070')

box(400,122,420,90)
txt(610,153,'患者多模态数据',16,True)
txt(610,186,'结构影像、代谢影像、临床量表等',11,color='#686868')

line([(610,212),(610,242),(550,242)])
arrow((550,242),(550,285))
line([(610,242),(970,242)])
arrow((970,242),(970,285))

box(400,285,300,88)
txt(550,312,'逐模态特征编码',14,True)
txt(550,349,'保留各模态的独立表示',11,color='#626262')

box(795,285,350,88)
txt(970,312,'逐模态建立患者关系图',14,True)
txt(970,349,'各模态分别确定固定候选邻居',11,color='#626262')

# The patient-only branch is independently fed by the input, not encoded tokens.
line([(400,167),(165,167),(165,850)])
arrow((165,850),(165,895))

# A quiet group boundary makes the core mechanism easy to locate.
ax.add_patch(FancyBboxPatch((350,437),795,704,
    boxstyle='round,pad=0,rounding_size=20',facecolor='none',
    edgecolor='#cfcfcf',linewidth=1,linestyle=(0,(4,4)),zorder=0))
txt(747.5,438,'群体信息传播分支',12,True,'#626262',
    bbox=dict(facecolor='white',edgecolor='none',pad=5))

arrow((550,373),(550,478))
arrow((970,373),(970,478))

box(390,478,720,125,True,'#f0f0f0')
txt(750,508,'创新点①：关系来源与传播内容分开',15,True,'#171717')
txt(750,547,'每一种模态内容，都在各模态关系图上传播',12)
txt(750,579,'例：用结构影像关系，传播代谢影像表示',11,color='#606060')

arrow((750,603),(750,656),'#454545',1.7)
box(390,656,720,180,True,'#f0f0f0')
txt(750,687,'创新点②：按内容调整传播方式',15,True,'#171717')
txt(750,731,'边权：关系相似度 ＋ 当前内容相似性',12)
txt(750,769,'变换：按内容模态学习发送／接收映射',12)
txt(750,808,'同一内容的映射在患者和关系来源之间共享',10.5,color='#606060')

arrow((750,836),(750,886),'#454545',1.7)
box(515,886,470,88)
txt(750,914,'不同关系的传播结果取平均',13,True)
txt(750,950,'残差更新各模态表示',11,color='#626262')

arrow((750,974),(750,1024))
box(515,1024,470,88)
txt(750,1052,'模态融合与类别读取',14,True)
txt(750,1088,'得到群体增强表示',11,color='#626262')

box(40,895,250,125)
txt(165,927,'自身信息分支',14,True)
txt(165,965,'仅利用患者自身',11)
txt(165,994,'的多模态信息',11)

arrow((750,1112),(750,1222))
line([(165,1020),(165,1267),(350,1267)])
arrow((350,1267),(390,1267))
box(390,1222,720,90)
txt(750,1251,'自身表示 ＋ 群体增强表示',15,True)
txt(750,1287,'相加后得到患者表示',11,color='#626262')

arrow((750,1312),(750,1392))
box(520,1392,460,90)
txt(750,1422,'多层感知机分类',15,True)
txt(750,1459,'输出诊断结果',12)

txt(600,1514,'加粗框突出两项核心设计；图中省略门控、具体网络层数和训练损失。',9.5,color='#737373')
path=OUT/'framework_c_blackwhite_cn.png'
fig.savefig(path,dpi=180,facecolor='white',metadata={'Title':'框架 C 中文框架图'})
plt.close(fig)
print(path)
