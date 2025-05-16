import numpy as np
import pickle
from glob import glob
import sys, re
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from help_fcts import trans_x

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

def to_low_mean_upp(r2_dict, data,alg, tr_te, sigmas):
  r2_mean=[]
  r2_low=[]
  r2_upp=[]
  for sigma in sigmas:
    trim_r2=np.sort(np.array(r2_dict[data][alg][tr_te][sigma]))[5:95]
    r2_mean.append(np.mean(trim_r2))
    r2_low.append(np.quantile(trim_r2,0.05))
    r2_upp.append(np.quantile(trim_r2,0.95))
  return r2_low, r2_mean, r2_upp

data_sets=['compactiv','power','super','temp']
data_titles2=['CPU Run Time', 'Tetouan Power\nConsumption', 'Superconductor\nTemperature', 'U.K.\nTemperature', 'Synthetic']
nus=[100, 0.5,1.5,2.5,10]
nu_titles=['$\\nu=\\infty$ (Gaussian)', '$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$','Cauchy']

lines=[Line2D([0],[0],color='C9',lw=3),Line2D([0],[0],color='C6',lw=3),Line2D([0],[0],color='C0',lw=3),Line2D([0],[0],color='C3',lw=3)]
labs = ['Training Error, Decreasing Bandwidth', 'Training Error, Constant Bandwidth', 'Test Error, Decreasing Bandwidth', 'Test Error, Constant Bandwidth']
fig1,axs1=plt.subplots(2,2,figsize=(10,4))
fig2,axs2=plt.subplots(4,4,figsize=(13,7))
axs=np.hstack((axs1.ravel(),axs2.T.ravel()))
ii=0
for nu, nu_title in zip(nus, nu_titles):
  r2_dict={}
  for data in data_sets:
    r2_dict[data]={}
    seeds=list(map(lambda s: s.split('_')[-1],glob('dd_data/dd_'+data+'_'+str(nu)+'_*')))
    for seed in seeds:
      fi=open('dd_data/dd_'+data+'_'+str(nu)+'_'+seed,'rb')
      r2_dict_seed=pickle.load(fi)
      fi.close()
      for alg in r2_dict_seed.keys():
        if not alg in r2_dict[data].keys():
          r2_dict[data][alg]={}
        for tr_te in r2_dict_seed[alg].keys():
          if not tr_te in r2_dict[data][alg].keys():
            r2_dict[data][alg][tr_te]={}
          for sig in r2_dict_seed[alg][tr_te].keys():
            if not sig in r2_dict[data][alg][tr_te].keys():
              r2_dict[data][alg][tr_te][sig]=[]
            r2_dict[data][alg][tr_te][sig].append(r2_dict_seed[alg][tr_te][sig])
  
  for data,data_title, data_title2 in zip(data_sets,data_titles,data_titles2):
    ax=axs[ii]
    sigmas=list(r2_dict[data]['krr']['tr'].keys())
    if max(sigmas)>10000:
      sigma_max=10000
    else:
      sigma_max=max(sigmas)
    sigma_min=max(min(sigmas),0.0)
    if nu==100:
      sigma_ticks=trans_x(np.linspace(trans_x(sigma_max), trans_x(sigma_min), 10),inv=True)
    else:
      sigma_ticks=trans_x(np.linspace(trans_x(sigma_max), trans_x(sigma_min), 5),inv=True)
    
    r2_tr_low_kgd, r2_tr_mean_kgd, r2_tr_upp_kgd = to_low_mean_upp(r2_dict, data, 'kgd', 'tr', sigmas)
    r2_te_low_kgd, r2_te_mean_kgd, r2_te_upp_kgd = to_low_mean_upp(r2_dict, data, 'kgd', 'te', sigmas)
    r2_tr_low_krr, r2_tr_mean_krr, r2_tr_upp_krr = to_low_mean_upp(r2_dict, data, 'krr', 'tr', sigmas)
    r2_te_low_krr, r2_te_mean_krr, r2_te_upp_krr = to_low_mean_upp(r2_dict, data, 'krr', 'te', sigmas)
    
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
    if ii in [2,3,7,11,15,19]:
      ax.set_xlabel('$\\sigma_m$')
    if ii in [0,2]:
      ax.set_ylabel('$1-R^2$')
    elif 4<=ii<=7:
      ax.set_ylabel(data_title2+'\n$1-R^2$')
    if ii<4:
      ax.set_title(data_title)
    if ii in [4,8,12,16]:
      ax.set_title(nu_title)
    
    ii+=1
    
    fig1.legend(lines, labs, loc='lower center', ncol=2)
    fig2.legend(lines, labs, loc='lower center', ncol=2)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.subplots_adjust(bottom=0.24)
    fig2.subplots_adjust(bottom=0.14)
  
    fig1.savefig('figures/dd_100.pdf')
    fig2.savefig('figures/dd_more.pdf')


