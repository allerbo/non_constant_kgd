import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys, re
sys.path.insert(1,'..')
from kgd import kgd
from help_fcts import r2, krr, cv10
from ffnn import  init_model, train_step, get_R2
from glob import glob
import pickle
from help_fcts import trans_x

def to_low_mean_upp(r2_dict,alg, tr_te, sigmas):
  r2_mean=[]
  r2_low=[]
  r2_upp=[]
  for sigma in sigmas:
    trim_r2=np.sort(np.array(r2_dict[alg][tr_te][sigma]))[5:95]
    r2_mean.append(np.mean(trim_r2))
    r2_low.append(np.quantile(trim_r2,0.05))
    r2_upp.append(np.quantile(trim_r2,0.95))
  return r2_low, r2_mean, r2_upp

labs = ['True Function', 'Noisy Observations','Predictions, Decreasing Bandwidth', 'Predictions, Constant Bandwidth', 'Training Error, Decreasing Bandwidth', 'Training Error, Constant Bandwidth', 'Test Error, Decreasing Bandwidth', 'Test Error, Constant Bandwidth']
lines=[Line2D([0],[0],color='C7',lw=3),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3),Line2D([0],[0],color='C9',lw=3),Line2D([0],[0],color='C6',lw=3),Line2D([0],[0],color='C0',lw=3),Line2D([0],[0],color='C3',lw=3)]
plt.cla()

def f_sin(x):
  y=np.sin(2*np.pi*x)
  return y

sigmas_m=np.array([1, 0.5, 0.1, 0.01])
LWS=[2.1,1.8,1.5,1.2]

n_tr=20
n_plot=300
lbda=1e-3


nu=100
seed=8

np.random.seed(seed)
x_tr=np.random.uniform(-1,1,n_tr).reshape((-1,1))
n_tr=x_tr.shape[0]
y_tr=f_sin(x_tr)+np.random.normal(0,.2,x_tr.shape)
x_te=np.linspace(-1,1,n_plot).reshape((-1,1))
y_te=f_sin(x_te)
x_tr_te=np.vstack((x_tr,x_te))
x_tr_te_argsort=x_tr_te.argsort(0)

nus =[100,0.5,1.5,2.5,10]
nu_titles=['$\\nu=\\infty$ (Gaussian)','$\\nu=1/2$\n(Laplace)','$\\nu=3/2$','$\\nu=5/2$', 'Cauchy']

fig1, axs1 = plt.subplot_mosaic([['a', 'b','c','d'], ['e', 'e','e','e']],figsize=(10,4))
fig2, axs2 = plt.subplot_mosaic([['a1', 'b1','c1','d1'], ['e1', 'e1','e1','e1'], ['a2', 'b2','c2','d2'], ['e2', 'e2','e2','e2'], ['a3', 'b3','c3','d3'], ['e3', 'e3','e3','e3'], ['a4', 'b4','c4','d4'], ['e4', 'e4','e4','e4'], ],figsize=(9,12))

for nu, nu_title, ax_suf in zip(nus,nu_titles, ['','1','2','3','4']):
  
  if nu==100:
    fig=fig1
    axs=axs1
  else:
    fig=fig2
    axs=axs2
    sigmas_m[0]=5
  
  for sigma_m, ax_key in zip(sigmas_m, ['a','b','c','d']):
    ax=axs[ax_key+ax_suf]
    _=ax.plot(x_te, y_te, 'C7', lw=LWS[0])
    _=ax.plot(x_tr,y_tr,'ok', ms=3)
    if nu==100:
      _=ax.text(-1,1.2,'$\\sigma_m='+str(sigma_m)+'$',fontsize=10)
    else:
      _=ax.text(-1,1.1,'$\\sigma_m='+str(sigma_m)+'$',fontsize=10)
    fh_krr=krr(x_tr_te,x_tr,y_tr,lbda,sigma_m, nu=nu)
    fh_kgd=kgd(x_tr_te,x_tr,y_tr, t_max=1/lbda, sigma_min=sigma_m, nu=nu)
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_krr[x_tr_te_argsort,0], 'C1',lw=LWS[1])
    ax.plot(x_tr_te[x_tr_te_argsort,0],fh_kgd[x_tr_te_argsort,0], 'C2',lw=LWS[2])
    ax.set_ylim([-1.5,1.5])
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_yticks([-1,0,1])
    ax.set_xticks([])
    ax.set_yticks([])
    if nu==100:
      fig.savefig('figures/syn_dd_100.pdf')
    else:
      if ax_key=='a':
        ax.set_ylabel(nu_title)
      fig.savefig('figures/syn_dd_more.pdf')
  
  data='syn'
  r2_dict={}
  seeds=list(map(lambda s: s.split('_')[-1],glob('dd_data/dd_'+data+'_'+str(nu)+'_*')))
  for seed in seeds:
    fi=open('dd_data/dd_'+data+'_'+str(nu)+'_'+seed,'rb')
    r2_dict_seed=pickle.load(fi)
    fi.close()
    for alg in r2_dict_seed.keys():
      if not alg in r2_dict.keys():
        r2_dict[alg]={}
      for tr_te in r2_dict_seed[alg].keys():
        if not tr_te in r2_dict[alg].keys():
          r2_dict[alg][tr_te]={}
        for sig in r2_dict_seed[alg][tr_te].keys():
          if not sig in r2_dict[alg][tr_te].keys():
            r2_dict[alg][tr_te][sig]=[]
          r2_dict[alg][tr_te][sig].append(r2_dict_seed[alg][tr_te][sig])
  
  ax=axs['e'+ax_suf]
  sigmas=list(r2_dict['krr']['tr'].keys())
  if max(sigmas)>10000:
    sigma_max=10000
  else:
    sigma_max=max(sigmas)
  sigma_min=max(min(sigmas),0.0)
  if nu==100:
    sigma_ticks=trans_x(np.linspace(trans_x(sigma_max), trans_x(sigma_min), 10),inv=True)
  else:
    sigma_ticks=trans_x(np.linspace(trans_x(sigma_max), trans_x(sigma_min), 5),inv=True)
  
  r2_tr_low_kgd, r2_tr_mean_kgd, r2_tr_upp_kgd = to_low_mean_upp(r2_dict, 'kgd', 'tr', sigmas)
  r2_te_low_kgd, r2_te_mean_kgd, r2_te_upp_kgd = to_low_mean_upp(r2_dict, 'kgd', 'te', sigmas)
  r2_tr_low_krr, r2_tr_mean_krr, r2_tr_upp_krr = to_low_mean_upp(r2_dict, 'krr', 'tr', sigmas)
  r2_te_low_krr, r2_te_mean_krr, r2_te_upp_krr = to_low_mean_upp(r2_dict, 'krr', 'te', sigmas)
  
  ax.cla()
  ax.plot(sigmas,1-np.array(r2_tr_mean_krr),'C6',lw=3)
  ax.plot(sigmas,1-np.array(r2_tr_low_krr),'C6--')
  ax.plot(sigmas,1-np.array(r2_tr_upp_krr),'C6--')
  
  ax.plot(sigmas,1-np.array(r2_tr_mean_kgd),'C9',lw=3)
  ax.plot(sigmas,1-np.array(r2_tr_low_kgd),'C9--')
  ax.plot(sigmas,1-np.array(r2_tr_upp_kgd),'C9--')
  
  ax.plot(sigmas,1-np.array(r2_te_mean_krr),'C3',lw=3)
  ax.plot(sigmas,1-np.array(r2_te_low_krr),'C3--')
  ax.plot(sigmas,1-np.array(r2_te_upp_krr),'C3--')
  
  ax.plot(sigmas,1-np.array(r2_te_mean_kgd),'C0',lw=3)
  ax.plot(sigmas,1-np.array(r2_te_low_kgd),'C0--')
  ax.plot(sigmas,1-np.array(r2_te_upp_kgd),'C0--')
  
  ax.set_xscale('function',functions=(lambda x:trans_x(x),lambda y:y))
  ax.set_xlim([sigmas[0],sigmas[-1]])
  
  
  sigma_labs=list(map(lambda s: f'{s:.2g}',sigma_ticks))
  sigma_labs=list(map(lambda s: s.replace('1e+04','10000'),sigma_labs))
  sigma_labs=list(map(lambda s: re.sub(r'(\d)\.(\d)e\+02',r'\1\2!0',s),sigma_labs))
  sigma_labs=list(map(lambda s: re.sub(r'(\d)e\+02',r'\1!00',s),sigma_labs))
  sigma_labs=list(map(lambda s: s.replace('!',''),sigma_labs))
  sigma_labs=list(map(lambda s: re.sub(r'0\.(\d)([^\d])',r'0.\1!0\2',s),sigma_labs))
  sigma_labs=list(map(lambda s: re.sub(r'^(\d)([^\d\.])',r'\1!.0\2',s),sigma_labs))
  sigma_labs=list(map(lambda s: re.sub(r'^(\d)$',r'\1!.0',s),sigma_labs))
  sigma_labs=list(map(lambda s: s.replace('!',''),sigma_labs))
  ax.set_xticks(sigma_ticks)
  ax.set_xticklabels(sigma_labs,fontsize=8.5)
  
  
  
  ax.set_ylim([-0.01,2.01])
  ax.axhline(1,color='k', ls='--')
  
  fig.legend(lines, labs, loc='lower center', ncol=4,fontsize=8)
  ax.set_xlabel('$\\sigma_m$')
  ax.set_ylabel('$1-R^2$')
  
  if nu==100:
    for sigma_m in sigmas_m:
      ax.axvline(sigma_m,color='black')
    fig.add_artist(plt.Line2D([0,1],[0.59,.59], transform=fig.transFigure,color='black'))
    fig.tight_layout()
    fig.subplots_adjust(bottom=.22)
    fig.savefig('figures/syn_dd_100.pdf')
  else:
    for yy in [0.27,0.515,0.755]:
      fig.add_artist(plt.Line2D([0,1],[yy,yy], transform=fig.transFigure,color='black'))
    fig.tight_layout()
    fig.subplots_adjust(bottom=.07)
    fig.savefig('figures/syn_dd_more.pdf')

