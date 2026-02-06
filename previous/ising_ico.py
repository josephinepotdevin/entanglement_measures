import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import  eigsh
from scipy.linalg import eig
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info.states import partial_trace
import matplotlib.pyplot as plt
import time
import json

def index_to_bitstring(s: int, N: int) -> str:
    # qubit 0 is the least significant bit, consistent with (s >> i) & 1
    return "".join(str((s >> i) & 1) for i in range(N))

def ket_string_from_state(psi: np.ndarray, N: int, tol: float = 1e-12, max_terms: int = 64) -> list[dict]:
    """
    Return a JSON-serializable list of nonzero amplitudes as kets.
    Each entry: {"amp_re":..., "amp_im":..., "ket":"|...>", "index":...}
    """
    terms = []
    nz = np.where(np.abs(psi) > tol)[0]
    # If it's large, keep only the biggest terms
    if nz.size > max_terms:
        nz = nz[np.argsort(-np.abs(psi[nz]))[:max_terms]]

    for s in nz:
        amp = psi[s]
        terms.append({
            "index": int(s),
            "ket": f"|{index_to_bitstring(int(s), N)}>",
            "amp_re": float(np.real(amp)),
            "amp_im": float(np.imag(amp)),
        })
    return terms


def log_negativity(dm, subsyst=[0]):
    X = dm.partial_transpose(subsyst).data
    # trace norm = sum of singular values
    s = np.linalg.svd(X, compute_uv=False)
    tr_norm = s.sum()
    return np.log2(tr_norm)

def log_negativity_pure_bipartition(psi, NA):
    """
    Log-negativity for a pure state |psi> across bipartition A|B,
    where A has NA qubits and B has N-NA qubits.
    """
    dimA = 1 << NA
    dimB = psi.size // dimA
    psi_mat = psi.reshape((dimA, dimB), order="F")  # qubit ordering consistency
    s = np.linalg.svd(psi_mat, compute_uv=False)
    return 2.0 * np.log2(np.sum(s))


def product_state_z(N, down_sites):
    """
    |0> = up (Z=+1), |1> = down (Z=-1)
    down_sites: iterable d'indices i (0..N-1) mis à |1>
    """
    dim = 1 << N
    psi = np.zeros(dim, dtype=np.complex128)

    s = 0
    for i in down_sites:
        s |= (1 << i)   # met le bit i à 1 => spin down sur i

    psi[s] = 1.0
    return psi


def product_half_up_half_down(N):
    assert N % 2 == 0

    dim = 1 << N
    psi = np.zeros(dim, dtype=np.complex128)

    # Example: first N/2 spins up (0), last N/2 spins down (1)
    s = 0
    for i in range(N//2, N):
        s |= (1 << i)

    psi[s] = 1.0
    return psi

def random_half_up_half_down(N, rng=np.random.default_rng()):
    assert N % 2 == 0

    bits = np.array([0]*(N//2) + [1]*(N//2))
    rng.shuffle(bits)

    s = sum(bits[i] << i for i in range(N))

    psi = np.zeros(1 << N, dtype=np.complex128)
    psi[s] = 1.0
    return psi


edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),
    (1,2),(1,5),(1,6),(1,7),
    (2,3),(2,7),(2,8),
    (3,4),(3,8),(3,9),
    (4,5),(4,9),(4,10),
    (5,6),(5,10),
    (6,7),(6,10),(6,11),
    (7,8),(7,11),
    (8,9),(8,11),
    (9,10),(9,11),
    (10,11)]

def ising_sparse(N, J=1.0, h=1.0):
    dim = 1 << N
    rows, cols, data = [], [], []
    # edges = [ #from chat
    #     (0,1),(0,4),(0,5),(0,7),(0,10),
    #     (1,2),(1,5),(1,6),(1,8),
    #     (2,3),(2,6),(2,7),(2,9),
    #     (3,4),(3,7),(3,8),(3,10),
    #     (4,5),(4,9),(4,11),
    #     (5,6),(5,11),
    #     (6,7),(6,10),
    #     (8,9),(8,10),(8,11),
    #     (9,10),(9,11),
    #     (7,9),
    # ]

    # --- Diagonal term: -h sum Z_i ---
    # In Z-basis: Z|0> = +|0>, Z|1> = -|1>
    for s in range(dim):
        e = 0.0
        for i in range(N):
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0
            e += -h * zi
        rows.append(s)
        cols.append(s)
        data.append(e)

    # --- Off-diagonal term: J sum X_i X_{i+1} ---
    # In Z-basis: X flips the bit. So X_i X_{i+1} flips bits i and i+1.
    for s in range(dim):
        for (i,j) in edges: #for icosahedral
            sp = s ^ (1 << i) ^ (1 << j)  # flip both neighboring spins
            rows.append(sp)
            cols.append(s)
            data.append(J)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H

if __name__ == '__main__':

#---------INITIALIZE--------------------------------------------------------------------------------------------------------------
    save=True
    v = 4
    name = 'Icosahedre benchmark negativity ICO1 initial state only 1vs1'
    N = 12
    J = 1.0
    h = 3.0
    tfin = 1.4
    tsteps = 100
    
    psi0 = product_state_z(12, down_sites=[0,2,6,10,11])
    print(psi0.shape)
    
    # psi0 = np.zeros(dim, dtype=np.complex128) #for chain
    # psi0[0] = 1.0    # |000...0>
    
    # psi0 = product_half_up_half_down(N)
#---------------------------------------------------------------------------------------------------------------------------------
    
    dimensions = np.full(N, 2)
    H = ising_sparse(N, J, h)
    start_time = time.perf_counter()

    t_ = np.linspace(0, tfin, tsteps)
    
    dim = 1 << N

    neg_01 = []
    neg_01_2 = []
    neg_01_23 = []
    neg_3v3 = []
    neg_6v6 = []
    for t in t_:
        print('Time t = ', t)

        psi_t = expm_multiply(-1j * H * t, psi0)
        psi_t = psi_t / np.linalg.norm(psi_t)

        # RDM_t_01 = partial_trace(DensityMatrix(psi_t), list(range(2,N)))
        RDM_t_01 = partial_trace(DensityMatrix(psi_t), [0,1,2,4,5,6,7,8,10,11]) # Like paper, take neighbours 3 and 9
        neg_t_01 = log_negativity(RDM_t_01)
        neg_01.append(neg_t_01)

        RDM_t_012 = partial_trace(DensityMatrix(psi_t), [0,1,2,4,5,6,7,10,11])
        neg_t_01_2 = log_negativity(RDM_t_012, [0,1])
        neg_01_2.append(neg_t_01_2)
        
        RDM_t_0123 = partial_trace(DensityMatrix(psi_t), [0,1,4,5,6,7,10,11])
        neg_t_01_23 = log_negativity(RDM_t_0123, [0,1])
        neg_01_23.append(neg_t_01_23)
        
        RDM_t_3v3 = partial_trace(DensityMatrix(psi_t), [0,1,4,5,6,10])
        neg_t_3v3 = log_negativity(RDM_t_3v3, [0,1,2])
        neg_3v3.append(neg_t_3v3)
        
        neg_t_6v6 = log_negativity_pure_bipartition(psi_t, NA=6)
        neg_6v6.append(neg_t_6v6)


    end_time = time.perf_counter()
    print("Time of computation: ", (end_time-start_time)/60, 'min')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['aquamarine','coral', 'springgreen','purple', 'dimgrey','gold']
    
    ax.plot(t_, neg_01, label=r'$\epsilon_{0|1}$',color=colors[0], marker='x', linestyle='-', linewidth=3)
    ax.plot(t_, neg_01_2, label=r'$\epsilon_{01|2}$',color=colors[1], marker='x', linestyle='-', linewidth=2)
    ax.plot(t_, neg_01_23, label=r'$\epsilon_{01|23}$',color=colors[2], marker='x', linestyle='-', linewidth=2)
    ax.plot(t_, neg_3v3, label=r'$\epsilon_{3v3}$',color=colors[3], marker='x', linestyle='-', linewidth=2)
    ax.plot(t_, neg_6v6, label=r'$\epsilon_{6v6}$',color=colors[4], marker='x', linestyle='-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Logarithmic negativity $\epsilon$')
    ax.set_title(f'Ising quantum quench h={h}, J={J} - chain {N} spins')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    plt.show()    
    
    fig.savefig(f'icosahedre{v}.png', dpi=300, bbox_inches = 'tight')
    
    if save:    
        
        data_to_dump = [{
            "Info:": name,
            "Number spin chain": N,
            "Computation time [min]": (end_time-start_time)/60,
            "Ising J": J,
            "Ising h": h,
            "Initial state (ket terms)": ket_string_from_state(psi0, N),
            "Time": t_.tolist(),
            "Edges": edges,
            "Log Neg site 3 vs 9": neg_01,
            # "Log Neg site 01 vs 23": neg_01_23,
            # "Log Neg site 3v3": neg_3v3,
            # "Log Neg site 6v6": neg_6v6
        }]
        with open(f'icosahedre{v}.json', 'w') as f:
            json.dump(data_to_dump, f)