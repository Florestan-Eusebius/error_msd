import numpy as np 
import TdistillationKet
import json

fid_list = [1, 0.995, 0.99, 0.985, 0.98]
epsilon_list = np.linspace(0, 0.1, 21)

data_file = open('Trate.dat', 'w+')
data_file.close()

data_file = open('Trate_var.dat', 'w+')
data_file.close()

data_file = open('Taccuracy.dat', 'w+')
data_file.close()

data_file = open('Taccuracy_var.dat', 'w+')
data_file.close()

N = 100

with open('Trate.dat', 'a') as rate_file, open('Trate_var.dat', 'a') as rate_var_file, open('Taccuracy.dat', 'a') as accuracy_file, open('Taccuracy_var.dat', 'a') as accuracy_var_file: 
    for fid in fid_list:
        p = 1 - np.sqrt(fid)
        rate = []
        rate_var = []
        accuracy = []
        accuracy_var = []
        for epsilon in epsilon_list:
            rt, v_rt, ac, v_ac = TdistillationKet.experiment(N, epsilon, p, p)
            rate.append(rt)
            rate_var.append(v_rt)
            accuracy.append(ac)
            accuracy_var.append(v_ac)
        print('finish fid =', fid)
        json.dump(rate, rate_file)
        rate_file.write('\n')
        json.dump(rate_var, rate_var_file)
        rate_var_file.write('\n')
        json.dump(accuracy, accuracy_file)
        accuracy_file.write('\n')
        json.dump(accuracy_var, accuracy_var_file)
        accuracy_var_file.write('\n')