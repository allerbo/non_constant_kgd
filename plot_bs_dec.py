import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from scipy.stats import wilcoxon

lines=[]
for c in ['C2','C0']: lines.append(Line2D([0],[0],color=c,lw=5))
labs=['p-value < 0.05','p-value $\\geq$ 0.05']


days=range(1,367)

nus = [0.5, 1.5, 2.5, 100, 10]
titles=['$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$','$\\nu=\infty$ (Gaussian)', 'Cauchy']
fig,axs=plt.subplots(5,2,figsize=(10,10))
for ax_r, (nu,title) in enumerate(zip(nus,titles)):
  p_dict={'GCV':[],'MML':[]}
  for day in days:
    fi=open('dec_data/bs_dec_nu_seed_'+str(nu)+'_'+str(day)+'.pkl','rb')
    r2_dict=pickle.load(fi)
    fi.close()
    p_dict['GCV'].append(wilcoxon(r2_dict['kgd'], r2_dict['krr_cv'], alternative='greater')[1])
    p_dict['MML'].append(wilcoxon(r2_dict['kgd'], r2_dict['krr_lm'], alternative='greater')[1])

  for alg, ax in zip(['GCV','MML'],axs[ax_r,:]):
    ps=np.array(p_dict[alg])
    ps05=ps[ps<0.05]
    ps95=ps[ps>=0.05]
    
    ax.hist(ps05,bins=np.arange(0,1.01,0.05),color='C2')
    ax.hist(ps95,bins=np.arange(0,1.01,0.05),color='C0')
    ax.set_title(alg+', '+title)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([0,160])
    ax.set_xticks(np.arange(0,1.05,0.1))

fig.suptitle('Distribution of p-values')
fig.legend(lines, labs, loc='lower center', ncol=4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.07)
fig.savefig('figures/bs_dec.pdf')
