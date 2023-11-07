import numpy as np
import pickle
from scipy.stats import wilcoxon


seeds=range(100)
algs=['kgd','gcv','mml']
alg_titles=['KGD','GCV','MML']
datas =['lin_sin', 'two_freq']
nus = [0.5, 1.5, 2.5, 100, 10]
nu_titles=['\\makecell{$\\nu=1/2$\\\\(Laplace)}','$\\nu=3/2$','$\\nu=5/2$','\\makecell{$\\nu=\infty$\\\\(Gaussian)}', 'Cauchy']

tab_dict={'kgd':{},'gcv':{},'mml':{}}
for alg in algs:
  for data in datas:
    tab_dict[alg][data]={}
    for nu in nus:
      tab_dict[alg][data][nu]={}

for data in datas:
  for nu in nus:
    r2_dict={'kgd':[],'gcv':[],'mml':[]}
    for seed in seeds:
      fi=open('dec_data/'+data+'_dec_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','rb')
      r2_dict_seed=pickle.load(fi)
      fi.close()
      for alg in algs:
        r2_dict[alg].append(r2_dict_seed[alg])
    for alg in algs:
      med=np.nanmedian(r2_dict[alg])
      q1=np.nanquantile(r2_dict[alg],0.25)
      q3=np.nanquantile(r2_dict[alg],0.75)
      tab_dict[alg][data][nu]['medq']= f'${med:.2f},\\ ({q1:.2f}, {q3:.2f})$'
      if alg=='kgd':
        tab_dict[alg][data][nu]['p_val']= '$-$'
      else:
        p_val=wilcoxon(r2_dict['kgd'], r2_dict[alg], alternative='greater', nan_policy='omit')[1][0]
        p_str= f'{p_val:.3g}'
        p_str=p_str.replace('e','\\cdot 10^{').replace('{-0','{-')+'}'
        tab_dict[alg][data][nu]['p_val']= '$'+p_str+'$'

for nu,nu_title in zip(nus,nu_titles):
  print('\\hline')
  for alg,alg_title in zip(algs,alg_titles):
    nu_str='\\multirow{3}{*}{'+nu_title+'}' if alg=='kgd' else ''
    print(nu_str.ljust(52,' ')+' & ' + alg_title + ' & ' + tab_dict[alg]['lin_sin'][nu]['medq'].ljust(21,' ') + ' & ' + tab_dict[alg]['lin_sin'][nu]['p_val'].ljust(10,' ')
                              + ' & ' + tab_dict[alg]['two_freq'][nu]['medq'].ljust(22,' ') + ' & ' + tab_dict[alg]['two_freq'][nu]['p_val'].ljust(10,' ')+' \\\\')

