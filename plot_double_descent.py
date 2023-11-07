import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from help_fcts import trans_x
import pickle


def to_low_med_upp(data_dict, kxx, tr_te, sigmas,month=None):
  r2_med=[]
  r2_low=[]
  r2_upp=[]
  month=None
  for sigma in sigmas:
    r2_med.append(np.median(np.array(data_dict[kxx][tr_te][sigma])))
    r2_low.append(np.quantile(np.array(data_dict[kxx][tr_te][sigma]),0.25))
    r2_upp.append(np.quantile(np.array(data_dict[kxx][tr_te][sigma]),0.75))
  return r2_low, r2_med, r2_upp


nus = [0.5, 1.5, 2.5, 100, 10]
titles=['$\\nu=1/2$ (Laplace)','$\\nu=3/2$','$\\nu=5/2$','$\\nu=\infty$ (Gaussian)', 'Cauchy']

sigma_ticks=trans_x(np.linspace(trans_x(10), trans_x(0.01), 10),inv=True)

sigmas_expl=np.array([5, 0.5, 0.1, 0.01])

fig,axs=plt.subplots(5,1,figsize=(10,10))
labs = ['Training Error, Decreasing Bandwidth', 'Training Error, Constant Bandwidth', 'Test Error, Decreasing Bandwidth', 'Test Error, Constant Bandwidth']
lines=[Line2D([0],[0],color='C0',lw=3),Line2D([0],[0],color='C4',lw=3),Line2D([0],[0],color='C2',lw=3),Line2D([0],[0],color='C1',lw=3)]

seeds=list(range(100))
nu=100
data='syn'
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if data=='syn':
  seeds=list(range(100))
elif data=='bs':
  seeds=list(range(1,367))

for nu,ax,title in zip(nus,axs,titles):
  sigmas=None
  data_dict={'krr': {'tr':{},'te':{}},'kgd': {'tr':{},'te':{}}}
  for seed in seeds:
    fi=open('dd_data/dd_'+data+'_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','rb')
    data_dict_seed=pickle.load(fi)
    fi.close()
    if sigmas is None:
      sigmas=list(data_dict_seed['krr']['tr'].keys())
      for sigma in sigmas:
        data_dict['krr']['tr'][sigma]=[]
        data_dict['krr']['te'][sigma]=[]
        data_dict['kgd']['tr'][sigma]=[]
        data_dict['kgd']['te'][sigma]=[]
    for sigma in sigmas:
      data_dict['krr']['tr'][sigma].append(data_dict_seed['krr']['tr'][sigma])
      data_dict['krr']['te'][sigma].append(data_dict_seed['krr']['te'][sigma])
      data_dict['kgd']['tr'][sigma].append(data_dict_seed['kgd']['tr'][sigma])
      data_dict['kgd']['te'][sigma].append(data_dict_seed['kgd']['te'][sigma])
  
  r2_tr_low_kgd, r2_tr_med_kgd, r2_tr_upp_kgd=to_low_med_upp(data_dict, 'kgd', 'tr', sigmas)
  r2_te_low_kgd, r2_te_med_kgd, r2_te_upp_kgd=to_low_med_upp(data_dict, 'kgd', 'te', sigmas)
  r2_tr_low_krr, r2_tr_med_krr, r2_tr_upp_krr=to_low_med_upp(data_dict, 'krr', 'tr', sigmas)
  r2_te_low_krr, r2_te_med_krr, r2_te_upp_krr=to_low_med_upp(data_dict, 'krr', 'te', sigmas)
  
  ax.cla()
  ax.plot(sigmas,1-np.array(r2_tr_med_krr),'C4',lw=3)
  ax.plot(sigmas,1-np.array(r2_tr_low_krr),'C4:')
  ax.plot(sigmas,1-np.array(r2_tr_upp_krr),'C4:')
  
  ax.plot(sigmas,1-np.array(r2_tr_med_kgd),'C0',lw=3)
  ax.plot(sigmas,1-np.array(r2_tr_low_kgd),'C0:')
  ax.plot(sigmas,1-np.array(r2_tr_upp_kgd),'C0:')
  
  ax.plot(sigmas,1-np.array(r2_te_med_krr),'C1',lw=3)
  ax.plot(sigmas,1-np.array(r2_te_low_krr),'C1:')
  ax.plot(sigmas,1-np.array(r2_te_upp_krr),'C1:')
  
  ax.plot(sigmas,1-np.array(r2_te_med_kgd),'C2',lw=3)
  ax.plot(sigmas,1-np.array(r2_te_low_kgd),'C2:')
  ax.plot(sigmas,1-np.array(r2_te_upp_kgd),'C2:')

  if data=='syn':
    for jj, sigma in enumerate(sigmas_expl):
      ax.axvline(sigma,color='k')
  
  
  ax.set_xscale('function',functions=(lambda x:trans_x(x),lambda y:y))
  #ax.set_xscale('function',functions=(lambda x:-np.log(x+1e-10),lambda y:y))
  ax.set_xlim([sigmas[0],sigmas[-1]])
  ax.set_xticks(sigma_ticks)
  ax.set_ylim([-0.01,1.01])
  ax.set_xlabel('$\\sigma_m$',fontsize=12)
  ax.set_ylabel('$1-R^2$', fontsize=12)
  ax.set_title(title)
  
  fig.legend(lines, labs, loc='lower center', ncol=2)
  fig.tight_layout()
  fig.subplots_adjust(bottom=.1)
  fig.savefig('figures/double_descent_'+data+'.pdf')

