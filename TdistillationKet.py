import numpy as np

GATE_I = np.identity(2)
GATE_Z = np.array([[1, 0], [0, -1]])
GATE_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
GATE_X = np.array([[0, 1], [1, 0]])
CTRL0 = np.array([[1, 0], [0, 0]])
CTRL1 = np.array([[0, 0], [0, 1]])
GATE_CZ = np.tensordot(CTRL0, GATE_I, 0) + np.tensordot(CTRL1, GATE_Z, 0)

def add_qubit(ket, ket_add):
    ket_new = np.tensordot(ket, ket_add, 0)
    return ket_new


def gate_on_site(gate, site, ket):
    if isinstance(site, int):
        site = [site]
    n = len(site)
    N = len(ket.shape)

    gate_ind = [2 * i + 1 for i in range(n)]

    gate_ket = np.tensordot(ket, gate, (site, gate_ind))
    new_ind = []
    ind = 0
    for i in range(N):
        if i in site:
            new_ind.append(N - n + site.index(i))
            ind += 1
        else:
            new_ind.append(i - ind)
    gate_ket = np.transpose(gate_ket, new_ind)
    return gate_ket

def tensor_product(operators):
    op = operators[0]
    for i in range(1,len(operators)):
        op = add_qubit(op,operators[i])
    return op

def rearrange_qubits(ket, sites_old, sites_new):
    N = len(ket.shape)
    new_ind = [] 
    for i in range(N):
        if i in sites_old:
            new_ind.append(sites_new[sites_old.index(i)])
        else:
            new_ind.append(i)
    ket = np.transpose(ket, new_ind)
    return ket

def swap_qubits(ket, site1, site2):
    return rearrange_qubits(ket, [site1, site2], [site2, site1])
    
def discard_qubit(ket, sites):
    if isinstance(sites, int):
        sites = [sites]
    N = len(ket.shape)
    n = len(sites)
    ind_dscd = []
    ind_prsv = []
    for i in range(N):
        if i in sites:
            ind_dscd.append(i)
        else:
            ind_prsv.append(i)
    ket = np.transpose(ket, ind_dscd + ind_prsv)
    ket = np.reshape(ket, (1<<len(ind_dscd), 1<<len(ind_prsv)))
    dscd, spctr, prsv = np.linalg.svd(ket, full_matrices=False)
    wgt = spctr * spctr.conj()
    k = wgt.shape[0]
    shape = (k,) + tuple(2 for i in range(N-n))
    prsv = prsv.reshape(shape)
    return wgt, prsv


def rotation_Z(theta):
    gate = np.array([[1,0],[0,np.exp(-1j*theta)]])
    return gate

def rotation_X(theta):
    gate = rotation_Z(theta)
    gate = np.matmul(GATE_H, gate)
    gate = np.matmul(gate, GATE_H)
    return gate

def noisy_CZ_channel(state, site, p):
    if not p == 0:
        sigma = np.sqrt(-2*np.log(1-2*p))
        theta1 = np.random.normal(0, sigma)
        theta2 = np.random.normal(0, sigma)
        errorZ1 = rotation_Z(theta1)
        errorZ2 = rotation_Z(theta2)
        state = gate_on_site(errorZ1, site[0], state)
        state = gate_on_site(errorZ2, site[1], state)
    state = gate_on_site(GATE_CZ, site, state)
    return state

def noisy_T_dag_channel(state, site, epsilon, p):
    MZ_1 = (GATE_I-GATE_Z) / np.sqrt(2)
    theta = 0
    if not epsilon == 0:
        sigma = np.sqrt(-2*np.log(1-2*epsilon))
        theta = np.random.normal(0, sigma)
    ancilla = np.array([1, np.exp(1j*(np.pi*1/4+theta))])/np.sqrt(2)
    ket = tensor_product([ancilla, state])
    ket = gate_on_site(GATE_H, site + 1, ket)
    ket = noisy_CZ_channel(ket, [0, site + 1], p )
    ket = gate_on_site(GATE_H, site + 1, ket)
    ket = swap_qubits(ket, 0, site + 1)
    ket = gate_on_site(MZ_1, 0, ket)
    ket = gate_on_site(GATE_X, site + 1, ket)
    wgt, ket = discard_qubit(ket, 0)
    return ket[0]

def T_distillation(epsilon, p, q):
    Xbasis = np.array([1, 1]) / np.sqrt(2)
    ket = tensor_product([Xbasis]*16)

    X_stabilizer = tensor_product([GATE_X]*8)
    Id = tensor_product([GATE_I]*8)
    X_stabilizer_proj = (Id+X_stabilizer)/2

    X_other = tensor_product([GATE_X]*7)
    Id = tensor_product([GATE_I]*7)
    X_other_proj_0 = (Id+X_other)/2
    X_other_proj_1 = (Id-X_other)/2

    ket = noisy_CZ_channel(ket, [0, 15], q)
    ket = noisy_CZ_channel(ket, [8, 9], p)
    ket = noisy_CZ_channel(ket, [8, 10], p)
    ket = noisy_CZ_channel(ket, [8, 11], p)
    ket = noisy_CZ_channel(ket, [8, 12], p)
    ket = noisy_CZ_channel(ket, [8, 13], p)
    ket = noisy_CZ_channel(ket, [8, 14], p)
    ket = noisy_CZ_channel(ket, [8, 15], q)
    ket = noisy_CZ_channel(ket, [4, 5], p)
    ket = noisy_CZ_channel(ket, [4, 6], p)
    ket = noisy_CZ_channel(ket, [4, 7], p)
    ket = noisy_CZ_channel(ket, [4, 12], p)
    ket = noisy_CZ_channel(ket, [4, 13], p)
    ket = noisy_CZ_channel(ket, [4, 14], p)
    ket = noisy_CZ_channel(ket, [4, 15], q)
    ket = noisy_CZ_channel(ket, [2, 3], p)
    ket = noisy_CZ_channel(ket, [2, 6], p)
    ket = noisy_CZ_channel(ket, [2, 7], p)
    ket = noisy_CZ_channel(ket, [2, 10], p)
    ket = noisy_CZ_channel(ket, [2, 11], p)
    ket = noisy_CZ_channel(ket, [2, 14], p)
    ket = noisy_CZ_channel(ket, [2, 15], q)
    ket = noisy_CZ_channel(ket, [1, 3], p)
    ket = noisy_CZ_channel(ket, [1, 5], p)
    ket = noisy_CZ_channel(ket, [1, 7], p)
    ket = noisy_CZ_channel(ket, [1, 9], p)
    ket = noisy_CZ_channel(ket, [1, 11], p)
    ket = noisy_CZ_channel(ket, [1, 13], p)
    ket = noisy_CZ_channel(ket, [1, 15], q)
    ket = gate_on_site(GATE_H, 15, ket)
    ket = noisy_CZ_channel(ket, [15, 3], p)
    ket = noisy_CZ_channel(ket, [15, 5], p)
    ket = noisy_CZ_channel(ket, [15, 6], p)
    ket = noisy_CZ_channel(ket, [15, 9], p)
    ket = noisy_CZ_channel(ket, [15, 10], p)
    ket = noisy_CZ_channel(ket, [15, 12], p)
    ket = gate_on_site(GATE_H, 3, ket)
    ket = gate_on_site(GATE_H, 5, ket)
    ket = gate_on_site(GATE_H, 6, ket)
    ket = gate_on_site(GATE_H, 7, ket)
    ket = gate_on_site(GATE_H, 9, ket)
    ket = gate_on_site(GATE_H, 10, ket)
    ket = gate_on_site(GATE_H, 11, ket)
    ket = gate_on_site(GATE_H, 12, ket)
    ket = gate_on_site(GATE_H, 13, ket)
    ket = gate_on_site(GATE_H, 14, ket)
    for i in range(15):
        ket = noisy_T_dag_channel(ket, i+1, epsilon, p)
    ket = gate_on_site(X_stabilizer_proj,[4,5,6,7,8,9,10,11],ket)
    ket = gate_on_site(X_stabilizer_proj,[1,2,3,4,5,6,7,15],ket)
    ket = gate_on_site(X_stabilizer_proj,[2,3,4,5,10,11,12,13],ket)
    ket = gate_on_site(X_stabilizer_proj,[1,2,5,6,9,10,13,14],ket)
    ket_0 = gate_on_site(X_other_proj_0,[8,9,10,11,12,13,14],ket)
    ket_1 = gate_on_site(X_other_proj_1,[8,9,10,11,12,13,14],ket)
    wgt_0, ket_0 = discard_qubit(ket_0, list(range(1,16)))
    wgt_1, ket_1 = discard_qubit(ket_1, list(range(1,16)))
    for i in range(2):
        ket_1[i] = gate_on_site(GATE_Z, 0, ket_1[i])

    return wgt_0, ket_0, wgt_1, ket_1

def T_distillation_flag(epsilon, p, q):
    Xbasis = np.array([1, 1]) / np.sqrt(2)
    ket = tensor_product([Xbasis]*17)

    X_stabilizer = tensor_product([GATE_X]*8)
    Id = tensor_product([GATE_I]*8)
    X_stabilizer_proj = (Id+X_stabilizer)/2

    X_other = tensor_product([GATE_X]*7)
    Id = tensor_product([GATE_I]*7)
    X_other_proj_0 = (Id+X_other)/2
    X_other_proj_1 = (Id-X_other)/2

    ket = noisy_CZ_channel(ket, [0, 15], q)
    ket = gate_on_site(GATE_H, 15, ket)
    ket = noisy_CZ_channel(ket, [15, 16], q)
    ket = gate_on_site(GATE_H, 15, ket)
    ket = noisy_CZ_channel(ket, [8, 15], q)
    ket = noisy_CZ_channel(ket, [4, 15], q)
    ket = noisy_CZ_channel(ket, [2, 15], q)
    ket = noisy_CZ_channel(ket, [1, 15], q)
    ket = noisy_CZ_channel(ket, [8, 16], q)
    ket = noisy_CZ_channel(ket, [4, 16], q)
    ket = noisy_CZ_channel(ket, [2, 16], q)
    ket = noisy_CZ_channel(ket, [1, 16], q)
    ket = gate_on_site(GATE_H, 15, ket)
    ket = noisy_CZ_channel(ket, [15, 16], q)
    ket = gate_on_site((GATE_X+GATE_I)/2, 16, ket)
    wgt, ket = discard_qubit(ket, 16)
    ket = ket[0]
    ket = noisy_CZ_channel(ket, [8, 9], p)
    ket = noisy_CZ_channel(ket, [8, 10], p)
    ket = noisy_CZ_channel(ket, [8, 11], p)
    ket = noisy_CZ_channel(ket, [8, 12], p)
    ket = noisy_CZ_channel(ket, [8, 13], p)
    ket = noisy_CZ_channel(ket, [8, 14], p)
    ket = noisy_CZ_channel(ket, [4, 5], p)
    ket = noisy_CZ_channel(ket, [4, 6], p)
    ket = noisy_CZ_channel(ket, [4, 7], p)
    ket = noisy_CZ_channel(ket, [4, 12], p)
    ket = noisy_CZ_channel(ket, [4, 13], p)
    ket = noisy_CZ_channel(ket, [4, 14], p)
    ket = noisy_CZ_channel(ket, [2, 3], p)
    ket = noisy_CZ_channel(ket, [2, 6], p)
    ket = noisy_CZ_channel(ket, [2, 7], p)
    ket = noisy_CZ_channel(ket, [2, 10], p)
    ket = noisy_CZ_channel(ket, [2, 11], p)
    ket = noisy_CZ_channel(ket, [2, 14], p)
    ket = noisy_CZ_channel(ket, [1, 3], p)
    ket = noisy_CZ_channel(ket, [1, 5], p)
    ket = noisy_CZ_channel(ket, [1, 7], p)
    ket = noisy_CZ_channel(ket, [1, 9], p)
    ket = noisy_CZ_channel(ket, [1, 11], p)
    ket = noisy_CZ_channel(ket, [1, 13], p)
    ket = noisy_CZ_channel(ket, [15, 3], p)
    ket = noisy_CZ_channel(ket, [15, 5], p)
    ket = noisy_CZ_channel(ket, [15, 6], p)
    ket = noisy_CZ_channel(ket, [15, 9], p)
    ket = noisy_CZ_channel(ket, [15, 10], p)
    ket = noisy_CZ_channel(ket, [15, 12], p)
    ket = gate_on_site(GATE_H, 3, ket)
    ket = gate_on_site(GATE_H, 5, ket)
    ket = gate_on_site(GATE_H, 6, ket)
    ket = gate_on_site(GATE_H, 7, ket)
    ket = gate_on_site(GATE_H, 9, ket)
    ket = gate_on_site(GATE_H, 10, ket)
    ket = gate_on_site(GATE_H, 11, ket)
    ket = gate_on_site(GATE_H, 12, ket)
    ket = gate_on_site(GATE_H, 13, ket)
    ket = gate_on_site(GATE_H, 14, ket)
    for i in range(15):
        ket = noisy_T_dag_channel(ket, i+1, epsilon, p)
    ket = gate_on_site(X_stabilizer_proj,[4,5,6,7,8,9,10,11],ket)
    ket = gate_on_site(X_stabilizer_proj,[1,2,3,4,5,6,7,15],ket)
    ket = gate_on_site(X_stabilizer_proj,[2,3,4,5,10,11,12,13],ket)
    ket = gate_on_site(X_stabilizer_proj,[1,2,5,6,9,10,13,14],ket)
    ket_0 = gate_on_site(X_other_proj_0,[8,9,10,11,12,13,14],ket)
    ket_1 = gate_on_site(X_other_proj_1,[8,9,10,11,12,13,14],ket)
    wgt_0, ket_0 = discard_qubit(ket_0, list(range(1,16)))
    wgt_1, ket_1 = discard_qubit(ket_1, list(range(1,16)))
    wgt_0 = wgt_0*wgt[0]
    wgt_1 = wgt_1*wgt[0]
    for i in range(2):
        ket_1[i] = gate_on_site(GATE_Z, 0, ket_1[i])

    return wgt_0, ket_0, wgt_1, ket_1

def fidelity(ket_0, ket_1):
    fid = np.abs(np.sum(ket_0*ket_1.conj()))**2
    return fid

def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def experiment(N, epsilon, p, q):
    W = []
    F = []
    ancilla = np.array([1, np.exp(1j*np.pi/4)])/np.sqrt(2)


    for i in range(N):
        wgt0, ket0, wgt1, ket1 = T_distillation(epsilon, p, q)
        for j in range(2):
            W.append(wgt0[j])
            W.append(wgt1[j])
            fd0 = fidelity(ket0[j], ancilla)
            fd1 = fidelity(ket1[j], ancilla)
            F.append(fd0)
            F.append(fd1)

    W = np.array(W)
    F = np.array(F)

    print(W)
    print(F)

    rate = np.sum(W) / N 
    var_r = np.std(W/N, ddof = 1) / np.sqrt(N)
    fid, var_f = weighted_avg_and_std(F, W)
    fid = 1 - fid
    var_f = var_f / np.sqrt(N-1)

    return rate, var_r, fid, var_f

def experiment_flag(N, epsilon, p, q):
    W = []
    F = []
    ancilla = np.array([1, np.exp(1j*np.pi/4)])/np.sqrt(2)


    for i in range(N):
        wgt0, ket0, wgt1, ket1 = T_distillation_flag(epsilon, p, q)
        for j in range(2):
            W.append(wgt0[j])
            W.append(wgt1[j])
            fd0 = fidelity(ket0[j], ancilla)
            fd1 = fidelity(ket1[j], ancilla)
            F.append(fd0)
            F.append(fd1)

    W = np.array(W)
    F = np.array(F)

    print(W)
    print(F)

    rate = np.sum(W) / N 
    var_r = np.std(W/N, ddof = 1) / np.sqrt(N)
    fid, var_f = weighted_avg_and_std(F, W)
    fid = 1 - fid
    var_f = var_f / np.sqrt(N-1)

    return rate, var_r, fid, var_f