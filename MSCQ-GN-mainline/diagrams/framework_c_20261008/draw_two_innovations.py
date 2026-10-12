from pathlib import Path
import os

OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.colors import to_rgba

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(14.8,7.2),dpi=200)
fig.patch.set_facecolor('white')
ax.set(xlim=(0,1480),ylim=(720,0),aspect='equal')
ax.axis('off')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
BLUE='#3684ad'; ORANGE='#c9833b'; INK='#262b30'; GRAY='#7d848a'

def text(x,y,s,size=12,bold=False,color=INK,ha='center',**kw):
    ax.text(x,y,s,ha=ha,va='center',fontsize=size,
        fontweight='bold' if bold else 'normal',color=color,linespacing=1.45,zorder=8,**kw)

def arrow(a,b,color=GRAY,lw=1.7,curve=0,scale=12):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',
        connectionstyle=f'arc3,rad={curve}',color=color,linewidth=lw,
        mutation_scale=scale,shrinkA=0,shrinkB=0,zorder=4))

def strip(cx,cy,color,w=55,h=18,pattern=(.3,.55,.8,.45)):
    x=cx-w/2; y=cy-h/2
    for k,a in enumerate(pattern):
        ax.add_patch(Rectangle((x+k*w/4,y),w/4,h,
            facecolor=to_rgba(color,a),edgecolor='white',lw=.7,zorder=6))
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,lw=.9,zorder=6))

def node(cx,cy,name,color):
    ax.add_patch(Circle((cx,cy),19,facecolor='white',edgecolor='#42494f',lw=1.3,zorder=5))
    text(cx,cy,name,10.5,True)
    strip(cx,cy+31,color,w=43,h=13)

def graph(cx,cy,color,relation):
    pts={'甲':(cx-73,cy-14),'乙':(cx+73,cy-14),'丙':(cx,cy+65)}
    edges=[('甲','乙'),('丙','乙')] if relation==0 else [('甲','乙'),('甲','丙')]
    for a,b in edges:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]],
                color='#8d9398',lw=1.6,zorder=1)
    for name,(x,y) in pts.items():node(x,y,name,color)
    # A separate curved colored arrow highlights content on an existing gray edge.
    arrow((cx-62,cy-35),(cx+62,cy-35),color,2.1,curve=-.45,scale=12)
    strip(cx,cy-68,color,w=44,h=14)

def person(cx,cy):
    ax.add_patch(Circle((cx,cy-9),9,facecolor='#d6d9dc',edgecolor='#535a60',lw=1))
    ax.add_patch(FancyBboxPatch((cx-18,cy+3),36,26,
        boxstyle='round,pad=0,rounding_size=9',facecolor='#e7e9eb',
        edgecolor='#535a60',lw=1))

def mapping(cx,cy,color,pattern):
    cell=25; size=50
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((cx-size/2+j*cell,cy-size/2+i*cell),cell,cell,
                facecolor=to_rgba(color,pattern[i*2+j]),edgecolor='white',lw=1.4,zorder=5))
    ax.add_patch(Rectangle((cx-size/2,cy-size/2),size,size,
        fill=False,edgecolor=color,lw=1.1,zorder=6))

text(740,37,'框架 C 的核心创新',21,True)
ax.plot([741,741],[91,658],color='#d7dadd',lw=1)

# Left: relationship source and message content are distinct axes.
text(362,104,'① 关系来源与传播内容分开',16,True)
text(362,143,'同一张关系图，可以传播不同模态内容',11,color='#666e75')
text(331,200,'结构影像内容',12,True,BLUE)
text(582,200,'代谢影像内容',12,True,ORANGE)
strip(331,228,BLUE,w=58,h=13)
strip(582,228,ORANGE,w=58,h=13)

text(95,316,'结构影像\n关系图',13,True)
text(95,512,'代谢影像\n关系图',13,True)
graph(331,318,BLUE,0)
graph(582,318,ORANGE,0)
graph(331,514,BLUE,1)
graph(582,514,ORANGE,1)
ax.plot([202,697],[429,429],color='#e1e3e5',lw=1,ls=(0,(3,4)))
text(366,635,'同一种内容，也可利用不同来源的关系',11.5,True)

# Right: two modality-specific message transformations on the same patient edge.
text(1108,104,'② 不同内容采用不同传播规则',16,True)
text(1108,143,'同一对患者，按模态分别处理消息',11,color='#666e75')
person(850,192); person(1370,192)
text(850,234,'患者甲',11,True)
text(1370,234,'患者乙',11,True)
text(1032,248,'发送变换',11.5,True)
text(1215,248,'接收变换',11.5,True)

for cy,color,name,ps,pt,lw in [
    (327,BLUE,'结构影像',(.2,.65,.85,.35),(.65,.25,.4,.8),1.6),
    (495,ORANGE,'代谢影像',(.7,.25,.35,.55),(.3,.75,.85,.2),2.5)]:
    text(850,cy-40,name+'内容',10.5,True,color)
    text(1370,cy-40,name+'内容',10.5,True,color)
    strip(850,cy,color,w=69,h=23)
    mapping(1032,cy,color,ps)
    mapping(1215,cy,color,pt)
    strip(1370,cy,color,w=69,h=23,pattern=(.5,.75,.4,.65))
    arrow((893,cy),(997,cy),color,1.8)
    arrow((1067,cy),(1180,cy),color,lw)
    arrow((1250,cy),(1327,cy),color,1.8)
    text(1124,cy-27,'边权',10,color=color)
    text(1124,cy+53,name+'使用自身的发送／接收变换',10.5,color=color)

text(1110,604,'边权由关系相似性与当前内容相似性共同确定',10.5,color='#555e65')
text(1110,638,'同一模态的变换，在患者与关系来源之间共享',10.5,color='#555e65')

strip(398,690,BLUE,w=25,h=10)
strip(433,690,ORANGE,w=25,h=10)
text(461,690,'色块：模态内容',10,color='#666e75',ha='left')
ax.plot([660,707],[690,690],color='#8d9398',lw=1.6)
text(721,690,'灰线：患者关系',10,color='#666e75',ha='left')
ax.add_patch(Circle((984,690),9,facecolor='white',edgecolor='#42494f',lw=1))
text(1007,690,'圆点：患者',10,color='#666e75',ha='left')

path=OUT/'framework_c_two_innovations.png'
fig.savefig(path,dpi=200,facecolor='white',metadata={'Title':'框架 C 的两项核心设计'})
plt.close(fig)
print(path)
