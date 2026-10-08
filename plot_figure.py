"""Plot only archived, audited results; no model fitting or outcome selection."""
from pathlib import Path
import argparse, hashlib, json, os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'run/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AUTHOR=os.environ.get('MANUSCRIPT_AUTHOR','')

def save(fig,base,data,alt):
    base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',metadata={'Author':AUTHOR,'Title':base.stem,'Creator':'Matplotlib','Subject':''})
    fig.savefig(base.with_suffix('.png'),dpi=450,bbox_inches='tight',metadata={'Author':AUTHOR,'Title':base.stem})
    fig.savefig(base.with_suffix('.jpg'),dpi=450,bbox_inches='tight',pil_kwargs={'quality':96})
    base.with_suffix('.json').write_text(json.dumps({'data':data,'alt_text':alt,'author':AUTHOR},ensure_ascii=False,indent=2)+'\n')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    p=json.loads((ROOT/'expected_results.json').read_text())
    rows=[x for x in p['paired_summaries'] if x['model_family']=='greedy_contextual_perceptron' and x['regime']=='news_plus_tweets']
    fig,axs=plt.subplots(1,2,figsize=(6.8,3.4),layout='constrained',sharey=True)
    data=[]
    for ax,field in zip(axs,['CorrectForm','FullForm']):
        rr=sorted([x for x in rows if x['field']==field],key=lambda x:x['seed'])
        for j,r in enumerate(rr):
            focal=r['net']; nf=r['nonfocal_exposures']['edited_correct']-r['nonfocal_exposures']['raw_correct'];total=focal+nf
            ax.plot([j-.17,j+.17],[focal,total],color='.45',linewidth=1)
            ax.scatter(j-.17,focal,marker='o',facecolors='white',edgecolors='black',zorder=3)
            ax.scatter(j+.17,total,marker='s',color='black',zorder=3)
            data.append({'field':field,'seed':r['seed'],'focal_net':focal,'nonfocal_net':nf,'total_net':total})
        ax.axhline(0,linestyle='--',color='.55',linewidth=.8);ax.set_xticks(range(3),[str(r['seed']) for r in rr]);ax.set_xlim(-.5,2.5);ax.set_xlabel('Semente');ax.set_title(field,fontsize=11)
    axs[0].set_ylabel('Saldo de decisões corretas');axs[0].set_yticks([-2,0,5,10,15]);axs[0].set_ylim(-2.5,17)
    axs[0].scatter([],[],marker='o',facecolors='white',edgecolors='black',label='Posição substituída');axs[0].scatter([],[],marker='s',color='black',label='Ensaio completo');axs[0].legend(loc='upper left',frameon=False,fontsize=8)
    save(fig,ROOT/'figures/efeito_focal_contextual',data,'No regime conjunto, CorrectForm tem saldos focais 10, 5 e 7 e saldos totais 7, menos 1 e 5. FullForm tem saldos focais 14, 15 e 15 e saldos totais 14, 14 e 14. Cada ponto representa uma semente fixa; as linhas ligam medidas da mesma execução e não são intervalos de confiança.')

if __name__=='__main__':main()
