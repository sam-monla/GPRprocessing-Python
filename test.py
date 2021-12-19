import numpy as np
import basic_proc as bproc
import file_mgmt as fm

header, data = fm.readDZT("/Users/Samuel/Documents/École/Université/Maitrise/St-Louis/Blandford_GPR_2017/GPR-Blandford-H2017/GPR-Blandford-H2017/GPR/GRID____086.3DS/FILE____001.DZT")

beg, end = bproc.rem_empty(data)
data = data[:,beg:end]

data1 = data[:100,:data.shape[1]//2]
print(data1.shape)
fdata,U,D,V = bproc.quick_SVD(data1,coeff=[0,None])

print(U.shape)
print(D.shape)
print(V.shape)