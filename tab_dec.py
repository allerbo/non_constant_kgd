import numpy as np
import pickle
from scipy.stats import wilcoxon
from glob import glob
import sys, re


DEC_DATA='dec_data/'


for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

data_titles={'compactiv': '\\makecell{CPU Run Time}', 'power': '\\makecell{Tetouan Power\\\\Consumption}', 'temp': '\\makecell{U.K.\\\\Temperature}', 'wood': '\\makecell{Aspen Fibres}', 'super': '\\makecell{Superconductor\\\\Critical\\\\Temperature}'}
alg_titles={'kgd0.02': 'KGD-D, 0.02', 'kgd0.05': 'KGD-D, 0.05', 'kgd0.1': 'KGD-D, 0.1', 'kgd0.2': 'KGD-D, 0.2',  'kgd0.5': 'KGD-D, 0.5', 'krr_gcv': 'KRR-GCV', 'krr_mml': 'KRR-MML'}
nu_titles={0.5: ', $\\nu=1/2$ (Laplace)', 1.5: ', $\\nu=3/2$', 2.5: ', $\\nu=5/2$', 10: ', Cauchy', 100: ''}

tab_dict={}
for nu in [0.5, 1.5, 2.5, 10, 100]:
  tab_dict[nu]={}
  nu_str='_'+str(nu)+'_'
  data_sets=sorted(list(set(map(lambda s: '_'.join(s.split('_')[2:3]),glob(DEC_DATA+'dec*'+nu_str+'*')))))
  r2_dict={}
  for data in data_sets:
    tab_dict[nu][data]={}
    r2_dict[data]={}
    seeds=list(map(lambda s: s.split('_')[-1],glob(DEC_DATA+'dec_'+data+nu_str+'*')))
    for seed in seeds:
      fi=open(DEC_DATA+'dec_'+data+nu_str+seed,'rb')
      r2_dict_seed=pickle.load(fi)
      fi.close()
      for alg in r2_dict_seed.keys():
        if not alg in r2_dict[data].keys():
          r2_dict[data][alg]=[]
        r2_dict[data][alg].append(r2_dict_seed[alg])
  
  for data in data_sets:
    for alg in r2_dict[data].keys():
      q1=np.nanquantile(r2_dict[data][alg],0.25)
      q2=np.nanquantile(r2_dict[data][alg],0.5)
      q3=np.nanquantile(r2_dict[data][alg],0.75)
      if alg in ['krr_gcv', 'krr_mml']:
        p_str='-'
      else:
        p_val_gcv=wilcoxon(r2_dict[data][alg], r2_dict[data]['krr_gcv'], alternative='greater', nan_policy='omit')[1]
        p_val_mml=wilcoxon(r2_dict[data][alg], r2_dict[data]['krr_mml'], alternative='greater', nan_policy='omit')[1]
        p_val=max(p_val_gcv, p_val_mml)
        p_str = re.sub(r'({-?)(0?)(\d+)',r'\1\3}',re.sub('e','\\\\cdot 10^{',f'{p_val:<7.2g}'))
        if p_val<0.05:
          p_str='\\bm{'+p_str+'}'
      tab_dict[nu][data][alg]=(f' & ${q2:#.3g},\\ ({q1:#.3g}, {q3:#.3g})$ & $'+ p_str +'$').replace('.,',',').replace('.)',')')


for nus in [[0.5, 1.5],[2.5,10],[100]]:
  seen_data=[]
  print('\\begin{table}')
  print('\\caption{'+str(nus)+'}')
  print('\\center')
  if len(nus)==1:
    print('\\begin{tabular}{|l|l|l|l|}')
    cl1='\\cline{3-4}'
    cl2='\\cline{2-4}'
  else:
    print('\\begin{tabular}{|l|l|l|l|l|l|}')
    cl1='\\cline{3-6}'
    cl2='\\cline{2-6}'
  print('\\hline')
  r2_col=''
  for nu in nus:
    r2_col+=' & \\multicolumn{2}{c|}{Test $R^2$'+nu_titles[nu]+'}'
  print('\\multirow{2}{*}{Data} & \\multirow{2}{*}{Method, $v_{R^2}$}'+r2_col+'\\\\')
  print(cl1)
  q_str='  &  '
  for nu in nus:
    q_str+=' & 50\\%,\\ \\ \\ \\ (25\\%,\\ 75\\%) & p-value'
  q_str+='\\\\'
  print(q_str)
  print('\\hline')
  for data in data_sets:
    print('\\multirow{7}{*}{'+data_titles[data.split('-')[0]]+'}')
    for alg in tab_dict[nus[0]][data].keys():
      if alg=='krr_gcv':
        print(cl2)
      alg_str=f'& {alg_titles[alg]:<11}'
      for nu in nus:
        alg_str+=tab_dict[nu][data][alg]
      print(alg_str+'\\\\')
    if alg=='krr_mml':
      print('\\hline')
  print('\\end{tabular}')
  print('\\label{tab:dec_'+str(nus[0])+'}')
  print('\\end{table}')
  print('\n\n')
  print('')
