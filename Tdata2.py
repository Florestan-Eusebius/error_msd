import numpy as np 
import TdistillationKet
import json

fid_q_list = np.linspace(0.995, 1, 11)
fid_p_list = [1, 0.995, 0.99, 0.985, 0.98]
epsilon = 0.02

data_file = open('Trate1.dat', 'w+')
data_file.close()

data_file = open('Trate_var1.dat', 'w+')
data_file.close()

data_file = open('Taccuracy1.dat', 'w+')
data_file.close()

data_file = open('Taccuracy_var1.dat', 'w+')
data_file.close()

N = 100

with open('Trate1.dat', 'a') as rate_file, open('Trate_var1.dat', 'a') as rate_var_file, open('Taccuracy1.dat', 'a') as accuracy_file, open('Taccuracy_var1.dat', 'a') as accuracy_var_file: 
    for fid_p in fid_p_list:
        p = 1 - np.sqrt(fid_p)
        rate = []
        rate_var = []
        accuracy = []
        accuracy_var = []
        for fid_q in fid_q_list:
            q = 1 - np.sqrt(fid_q)
            rt, v_rt, ac, v_ac = TdistillationKet.experiment(N, epsilon, p, q)
            rate.append(rt)
            rate_var.append(v_rt)
            accuracy.append(ac)
            accuracy_var.append(v_ac)
        print('finish fid =', fid_p)
        json.dump(rate, rate_file)
        rate_file.write('\n')
        json.dump(rate_var, rate_var_file)
        rate_var_file.write('\n')
        json.dump(accuracy, accuracy_file)
        accuracy_file.write('\n')
        json.dump(accuracy_var, accuracy_var_file)
        accuracy_var_file.write('\n')