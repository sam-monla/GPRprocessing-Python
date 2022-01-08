import numpy as np
import basic_proc as bproc
import file_mgmt as fm

a = [0,1,2,3,4,5]
b = 3

for elem in range(len(a)):
    if b == 3:
        for i in range(len(a)):
            print("allo")
    elif b == 2:
        for i in a:
            print(1)
    
    print(elem,i)
