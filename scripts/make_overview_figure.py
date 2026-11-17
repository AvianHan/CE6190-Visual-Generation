#!/usr/bin/env python3
"""Figure 1: vector overview, styled after the four user-provided draw-figs.
Real output thumbnails are only scaled. Run with matplotlib and Pillow installed.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
W,H=495,368
fig=plt.figure(figsize=(6.875,H/72)); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
INK='#211B17'; EDGE='#6B6B68'; BLUE='#3A607D'; PANEL='#E2E9EF'; BG='#F3F5F5'
C=['#B5B5B5','#466F87','#78B6C8','#F2D98E','#E48578']
def text(x,y,s,size=8,ha='center',bold=False,color=INK):
    return ax.text(x,y,s,fontsize=size,ha=ha,va='center',weight='bold' if bold else 'normal',color=color,linespacing=1.03,zorder=6)
def box(x,y,w,h,fill='white',edge=EDGE,r=4,lw=.95,ls='-'):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fill,ec=edge,lw=lw,ls=ls,zorder=2); ax.add_patch(p); return p
def path(pts,arrow=True,lw=1.2,color=EDGE,ls='-'):
    # One continuous stroke keeps elbows joined consistently.
    if arrow:
        route=MplPath(pts,[MplPath.MOVETO]+[MplPath.LINETO]*(len(pts)-1))
        ax.add_patch(FancyArrowPatch(path=route,arrowstyle='-|>',mutation_scale=8,
                                    lw=lw,color=color,linestyle=ls,
                                    joinstyle='round',capstyle='round',zorder=3))
    else:
        ax.plot(*zip(*pts),color=color,lw=lw,ls=ls,zorder=3,
                solid_joinstyle='round',solid_capstyle='round')
def grid(x,y,c,w=15,pattern=0):
    box(x,y,w,w,fill='#FFFFFF',edge='#8B9398',r=2,lw=.7)
    for i in range(3):
        for j in range(3):
            alpha=[.24,.5,.8][(i*2+j+pattern)%3]
            ax.add_patch(Rectangle((x+2+j*(w-4)/3,y+2+i*(w-4)/3),(w-4)/3-.5,(w-4)/3-.5,fc=matplotlib.colors.to_rgba(c,alpha),ec='none',zorder=4))
def dot(x,y,c,r=3): ax.add_patch(Circle((x,y),r,fc=c,ec=EDGE,lw=.6,zorder=4))
# Roomier layout: keep print-scale type and give each group its own whitespace.
box(10,6,475,82,fill=PANEL,edge='none',r=8)
text(22,20,'(a) Decision signals',10,ha='left',bold=True,color=BLUE)
xs=[57,152,247,342,437]
names=['Uncached','TeaCache','MagCache','SeaCache','SenCache']
labels=['always compute','input change','magnitude ratio','spectral change','sensitivity-weighted change']
for x,name,label,c in zip(xs,names,labels,C):
    text(x,39,name,9.5,bold=True)
    ax.plot([x-18,x+18],[45,45],color=c,lw=1.8,zorder=4,solid_capstyle='round')
    text(x,80,label,7.5)
# Schematic icons retain the original method-color mapping.
for j in range(4): box(37+j*11,54,8,13,fill=C[0],r=1,lw=.6)
grid(127,53,C[1],15); grid(162,53,C[1],15,pattern=1); path([(145,60.5),(159,60.5)])
for xx,heights in [(226,[9,15,11]),(251,[6,10,8])]:
    for j,h in enumerate(heights): ax.add_patch(Rectangle((xx+j*5,68-h),3,h,fc=C[2],ec=EDGE,lw=.4,zorder=4))
text(247,60,':',10)
t=np.linspace(0,1,90)
for off,amp in [(0,3.3),(8,2.2)]:
    ax.plot(320+44*t,56+off+amp*np.sin(4*np.pi*t),color='#9B833E',lw=.9,zorder=4)
grid(416,53,C[4],15); text(440,60.5,'×',10); dot(455,60.5,C[4],6)
text(455,60.5,'w',8,bold=True)
# Straight policy-to-rules-to-gate connection, clear of every title.
for x in xs: path([(x,88),(x,98)],False,lw=.9)
path([(57,98),(437,98)],False,lw=.9)
text(247.5,107,'select one',8)
box(10,116,325,140,fill=PANEL,edge='none',r=8)
box(342,116,143,140,fill=PANEL,edge='none',r=8)
text(22,235,'(b) Shared',10,ha='left',bold=True,color=BLUE)
text(22,248,'reuse step',10,ha='left',bold=True,color=BLUE)
text(352,129,'(c) Paired evaluation',10,ha='left',bold=True,color=BLUE)
box(18,128,112,43,fill='white',edge='#9CA4A9',r=4,lw=.8)
text(74,140,'Protection rules (e.g.)',8.5,bold=True)
text(74,153,'force selected steps to compute',7.5)
text(74,164,'at most $K$ reuses in a row',7.5)
path([(66.5,98),(66.5,128)])
path([(66.5,171),(66.5,190)])
text(24,204,r'$a_k$',10)
box(45,190,43,28,fill='white',r=4); text(66.5,204,'gate',9.5,bold=True)
path([(32,204),(45,204)])
text(120,207,'compute',8)
path([(88,198),(139,198),(139,155),(163,155)])
# Compute returns the current output and refreshes the residual cache.
for x,y in [(163,149),(169,145),(175,141)]: box(x,y,31,21,fill='white',r=3,lw=.9)
text(191,151.5,'F',10,bold=True)
text(191,130,'block stack',8)
path([(206,152),(253,152),(253,204)],arrow=False)
path([(191,162),(191,193)])
text(199,182,'refresh\nresidual',8,ha='left')
grid(180,193,'#79A4B6',22)
text(191,229,'residual cache',8)
# Reuse adds the cached residual to the current input.
path([(88,211),(105,211),(105,246),(225,246),(225,212)])
text(122,233,'reuse',8)
ax.add_patch(Circle((225,204),8,fc='white',ec=EDGE,lw=.95,zorder=4)); text(225,204,'+',12)
path([(202,204),(217,204)])
path([(233,204),(271,204)])
box(271,190,55,34,fill='white',r=4)
text(298.5,201,'output head',8.5,bold=True); text(298.5,214,'+ sampler',8.5)
text(298.5,239,'every step',8)
text(413.5,147,'same prompt + seed',8.5)
for x,method,filename in [(352,'TeaCache','teacache_0.5'),(423,'Uncached','uncached')]:
    im=Image.open(ROOT/f'report/figures/qualitative_inputs/flux_db_101_{filename}.png')
    ax.imshow(im,extent=(x,x+52,214,162),zorder=3,interpolation='antialiased')
    box(x,162,52,52,fill='none',edge=EDGE,r=2,lw=.9)
    text(x+26,225,method,8.5,bold=True)
path([(326,204),(352,204)])
path([(378,234),(378,237),(449,237),(449,234)],False,lw=.75)
text(413.5,248,'speedup · LPIPS',8)
box(10,270,475,77,fill='#F3F5F5',edge='none',r=7)
text(22,284,'(d) FLUX compute placement at similar speedup',10,ha='left',bold=True,color=BLUE)
# Real modal schedules, configuration-level summary metrics.
text(91,301,'step',7.5,ha='right')
start,pitch,cw=99,7.5,6.1
for k in [1,6,28]: text(start+(k-1)*pitch+cw/2,301,str(k),7.5)
text(337,301,'Computed',8); text(392,301,'Speedup',8); text(454,301,r'LPIPS $\downarrow$',8)
rows=[('TeaCache','1100100100100100100100100101',C[1],'2.45×','0.3231'),('MagCache','1110100100110000100000100101',C[2],'2.47×','0.1996'),('SenCache','1111110001000100010001000100',C[4],'2.46×','0.1576')]
for j,(name,s,c,speed,err) in enumerate(rows):
    assert len(s)==28 and s.count('1')==11
    y=315+j*13
    text(22,y,name,8.5,ha='left',bold=True)
    for k,b in enumerate(s):
        x=start+k*pitch
        ax.add_patch(Rectangle((x,y-3.7),cw,7.4,fc=c if b=='1' else 'white',ec=c if b=='1' else '#AEBBC3',lw=.5,zorder=4))
        if j==2 and k>=6 and b=='1': dot(x+cw/2,y,INK,1)
    text(337,y,'11 / 28',8.5); text(392,y,speed,8.5); text(454,y,err,8.5,bold=True)
text(247.5,359,'Filled: compute    Open: reuse    • Cap-forced refresh ($K=3$)',8)
OUT=ROOT/'report/figures/caching_overview'
for ext in ['pdf','svg','png']: fig.savefig(OUT.with_suffix('.'+ext),dpi=300,facecolor='white')
print(OUT.with_suffix('.pdf'))
