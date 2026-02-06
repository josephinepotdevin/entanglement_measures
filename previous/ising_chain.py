import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import  eigsh
from scipy.linalg import eig
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info.states import partial_trace
import matplotlib.pyplot as plt
import time
# from gen_gilbert_algo import gilbert
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

def entropy_von_neumann(density, subsyst=[0]):
    reduced_density = partial_trace(density, subsyst)
    eigenvalues = np.linalg.eigvalsh(reduced_density)
    nonzero = eigenvalues[eigenvalues > 0]
    
    return -np.sum(nonzero * np.log(nonzero) / np.log(2))

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

def ising_sparse(N, J=1.0, h=1.0):
    dim = 1 << N
    rows, cols, data = [], [], []

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
        for i in range(N): # for periodic BC put N on chains
            j = (i+1) % N # periodic boundary conditions on chains ATTENTION aussi changer à range(N)
            # j = i+1
            sp = s ^ (1 << i) ^ (1 << j)  # flip both neighboring spins
            rows.append(sp)
            cols.append(s)
            data.append(J)

    # for s in range(dim): # ESSAY POUR VOIR INTEGRABILITE VS REVIVAL TIMES
    #     for i in range(N):
    #         j = (i+2) % N 
    #         sp = s ^ (1 << i) ^ (1 << j)  
    #         rows.append(sp)
    #         cols.append(s)
    #         data.append(2*J)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H


def ising_sparse_v2(N, J=1.0, h=1.0, periodic=True):
    """
    Transverse-Field Ising Model (TFIM) in the Z-basis:
        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    Parameters
    ----------
    N : int
        Number of spins.
    J : float
        Ising coupling.
    h : float
        Transverse field strength.
    periodic : bool
        If True: periodic boundary conditions, else open.

    Returns
    -------
    H : csr_matrix (2^N x 2^N), complex128
    """
    dim = 1 << N
    rows, cols, data = [], [], []

    # --- Diagonal term: -J sum Z_i Z_{i+1} ---
    # Z|0> = +|0>, Z|1> = -|1>
    max_i = N if periodic else (N - 1)
    for s in range(dim):
        e = 0.0
        for i in range(max_i):
            j = (i + 1) % N
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0
            zj = 1.0 if ((s >> j) & 1) == 0 else -1.0
            e += -J * (zi * zj)
        rows.append(s)
        cols.append(s)
        data.append(e)

    # --- Off-diagonal term: -h sum X_i ---
    # X flips bit i: |...b_i...> -> |...1-b_i...>
    for s in range(dim):
        for i in range(N):
            sp = s ^ (1 << i)  # flip spin i
            rows.append(sp)
            cols.append(s)
            data.append(-h)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H


def ising_diag(H):
    # eig_values, eig_vectors = np.linalg.eig(H.toarray())
    eig_values, eig_vectors = eig(H.toarray())
    # eig_values, eig_vectors = eigsh(H, k = H.shape[0]-1)
    return eig_values, eig_vectors


if __name__ == '__main__':
#------------------------------------------------------------------------------------------------------------------
    v=00
    save=False
    N = 10
    t_stop_gilbert = 5
    J = -1.0
    h = 1.0
    tfin = 3
    tstep = 100
    name = 'chain 10 spin ini |1111101111> tout (0v1, 0v2, ..., 6v6, S) periodic BC change x-z in H'
#------------------------------------------------------------------------------------------------------------------
    
    dimensions = np.full(N, 2)
    start_time = time.perf_counter()
    # H = ising_sparse(N, J, h)
    H = ising_sparse_v2(N, J, h)
    
    # eig_values, eig_vectors = ising_diag(H)
    # P = eig_vectors
    # P_inv = np.linalg.inv(P)

    t_ = np.linspace(0, tfin, tstep)
    
    
    dim = 1 << N

    S_01 = []
    
    neg_01 = []
    neg_02 = []
    neg_03 = []
    neg_04 = []
    neg_05 = []
    neg_06 = []
    
    neg_012 = []
    neg_0_123 = []
    neg_01_23 = []
    neg_3v3 = []
    neg_6v6 = []
    dist = []
    
    # down_sites = list(range(N))          # all down
    # down_sites.remove(N//2)                # make site i0 = up
    # psi0 = product_state_z(N, down_sites)
        
    psi0 = np.zeros(dim, dtype=np.complex128) #for chain
    psi0[0] = 1.0    # |000...0>
    
    # psi0 = product_half_up_half_down(N)
    
    for t in t_:
        print('Time t = ', t)

        psi_t = expm_multiply(-1j * H * t, psi0)
        psi_t = psi_t / np.linalg.norm(psi_t)
        
        # psi_t = (P @ np.diag(np.exp(-1j*eig_values*t)) @ P_inv) @ psi0

        RDM_t_01 = partial_trace(DensityMatrix(psi_t), list(range(2,N)))
        # RDM_t_01 = partial_trace(DensityMatrix(psi_t), [2,3,4,5,6,7,8,9,10,11])
        neg_t_01 = log_negativity(RDM_t_01)
        neg_01.append(neg_t_01)
        
        # RDM_t_02 = partial_trace(DensityMatrix(psi_t), [1,3,4,5,6,7,8,9,10,11])
        # neg_t_02 = log_negativity(RDM_t_02)
        # neg_02.append(neg_t_02)
        
        # RDM_t_03 = partial_trace(DensityMatrix(psi_t), [1,2,4,5,6,7,8,9,10,11])
        # neg_t_03 = log_negativity(RDM_t_03)
        # neg_03.append(neg_t_03)
        
        # RDM_t_04 = partial_trace(DensityMatrix(psi_t), [1,2,3,5,6,7,8,9,10,11])
        # neg_t_04 = log_negativity(RDM_t_04)
        # neg_04.append(neg_t_04)
        
        # RDM_t_05 = partial_trace(DensityMatrix(psi_t), [1,2,3,4,6,7,8,9,10,11])
        # neg_t_05 = log_negativity(RDM_t_05)
        # neg_05.append(neg_t_05)
        
        # RDM_t_06 = partial_trace(DensityMatrix(psi_t), [1,2,3,4,5,7,8,9,10,11])
        # neg_t_06 = log_negativity(RDM_t_06)
        # neg_06.append(neg_t_06)

        RDM_t_012 = partial_trace(DensityMatrix(psi_t), list(range(3,N)))
        neg_t_012 = log_negativity(RDM_t_012)
        neg_012.append(neg_t_012)

        RDM_t_0123 = partial_trace(DensityMatrix(psi_t), list(range(4,N)))
        neg_t_0_123 = log_negativity(RDM_t_0123)
        neg_0_123.append(neg_t_0_123)

        neg_t_01_23 = log_negativity(RDM_t_0123, [0,1])
        neg_01_23.append(neg_t_01_23)
        
        # RDM_t_3v3 = partial_trace(DensityMatrix(psi_t), list(range(6,N)))
        # neg_t_3v3 = log_negativity(RDM_t_3v3, [0,1,2])
        # neg_3v3.append(neg_t_3v3)

        # neg_t_6v6 = log_negativity_pure_bipartition(psi_t, NA=6)
        # neg_6v6.append(neg_t_6v6)
        
        # S_t_01 = entropy_von_neumann(DensityMatrix(psi_t), [0,1,2,3,4,5])
        # S_01.append(S_t_01)
        
        # dist.append(gilbert(DensityMatrix(psi_t), dimensions, t_stop_gilbert)[0])    
        # dist.append(gilbert(RDM_t_012, [2,2,2], t_stop_gilbert)[0])    

    end_time = time.perf_counter()
    print("Time of computation: ", (end_time-start_time)/60, 'min')
    # plt.plot(t_, dist)
    # plt.tight_layout()
    # plt.show()  
    
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['aquamarine', 'coral', 'springgreen','purple', 'dimgrey','gold']
    
    ax.plot(t_, neg_01, label=r'$\epsilon_{0|1}$',color=colors[0], linestyle='-', linewidth=2)
    # ax.plot(t_, neg_02, label=r'$\epsilon_{0|2}$',color=colors[1], linestyle='-', linewidth=1)
    # ax.plot(t_, neg_03, label=r'$\epsilon_{0|3}$',color=colors[2], linestyle='-', linewidth=1)
    # ax.plot(t_, neg_04, label=r'$\epsilon_{0|4}$',color=colors[3], linestyle='-', linewidth=1)
    # ax.plot(t_, neg_05, label=r'$\epsilon_{0|5}$',color=colors[4], linestyle='-', linewidth=1)
    # ax.plot(t_, neg_06, label=r'$\epsilon_{0|6}$',color=colors[5], linestyle='-', linewidth=1)
    ax.plot(t_, neg_012, label=r'$\epsilon_{0|12}$',color=colors[1], marker='x', linestyle='-', linewidth=2)
    ax.plot(t_, neg_0_123, label=r'$\epsilon_{0|123}$',color=colors[2], marker='x', linestyle='-', linewidth=2)
    ax.plot(t_, neg_01_23, label=r'$\epsilon_{01|23}$',color=colors[3], marker='x', linestyle='-', linewidth=2)
    # ax.plot(t_, neg_3v3, label=r'$\epsilon_{3v3}$',color=colors[4], marker='x', linestyle='-', linewidth=2)
    # ax.plot(t_, neg_6v6, label=r'Logarithmic negativity $\epsilon_{6v6}$',color=colors[1], marker='x', linestyle='-', linewidth=2)
    # ax.plot(t_, S_01, label=r'Entropy of Von Neumann $S_{6v6}$',color=colors[0], marker='x', linestyle='-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Logarithmic negativity $\epsilon$')
    ax.set_title(f'Ising quantum quench h={h}, J={J} - chain {N} spins')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig.savefig(f'chain{v}_2spins.png', dpi=300, bbox_inches = 'tight')

    # fig2, ax2 = plt.subplots(figsize=(8, 5))
        
    # ax2.plot(t_, neg_01, label=r'$\epsilon_{0|1}$',color=colors[0], marker='x', linestyle='-', linewidth=2)
    # ax2.plot(t_, neg_012, label=r'$\epsilon_{0|12}$',color=colors[1], marker='x', linestyle='-', linewidth=2)
    # ax2.plot(t_, neg_0_123, label=r'$\epsilon_{0|123}$',color=colors[2], marker='x', linestyle='-', linewidth=2)
    # ax2.plot(t_, neg_01_23, label=r'$\epsilon_{01|23}$',color=colors[3], marker='x', linestyle='-', linewidth=2)
    # ax2.plot(t_, neg_3v3, label=r'$\epsilon_{3v3}$',color=colors[4], marker='x', linestyle='-', linewidth=2)
    # ax2.set_xlabel('Time [s]')
    # ax2.set_ylabel(r'Logarithmic negativity $\epsilon$')
    # ax2.set_title(f'Ising quantum quench h={h}, J={J} - chain {N} spins')
    # ax2.legend(fontsize=10)
    # ax2.grid(True, linestyle=':', alpha=0.7)
    
    # fig2.savefig(f'chain{v}_bigsites.png', dpi=300, bbox_inches = 'tight')
    
    # fig3, ax3 = plt.subplots(figsize=(8, 5))
        
    # ax3.plot(t_, neg_6v6, label=r'Logarithmic negativity $\epsilon_{6v6}$',color=colors[0], marker='x', linestyle='-', linewidth=2)
    # ax3.plot(t_, S_01, label=r'Entropy of Von Neumann $S_{6v6}$',color=colors[1], marker='x', linestyle='-', linewidth=2)
    # ax3.set_xlabel('Time [s]')
    # ax.set_ylabel('Bipartite entanglement measure')
    # ax3.set_title(f'Ising quantum quench h={h}, J={J} - chain {N} spins')
    # ax3.legend(fontsize=10)
    # ax3.grid(True, linestyle=':', alpha=0.7)
    
    # fig3.savefig(f'chain{v}_SvsE.png', dpi=300, bbox_inches = 'tight')
    
    plt.tight_layout()
    plt.show()    
    

    if save:
        data_to_dump = [{
            "Info:": name,
            "Number spin chain": N,
            "Computation time [min]": (end_time-start_time)/60,
            "Ising J": J,
            "Ising h": h,
            "Initial state (ket terms)": ket_string_from_state(psi0, N),
            "Time": t_.tolist(),
            "Log Neg site 0 vs 1": neg_01,
            "Log Neg site 0 vs 2": neg_02,
            "Log Neg site 0 vs 3": neg_03,
            "Log Neg site 0 vs 4": neg_04,
            "Log Neg site 0 vs 5": neg_05,
            "Log Neg site 0 vs 6": neg_06,
            "Log Neg site 0 vs 12": neg_012,
            "Log Neg site 0 vs 123": neg_0_123,
            "Log Neg site 01 vs 23": neg_01_23,
            "Log Neg site 3v3": neg_3v3,
            "Log Neg site 6v6": neg_6v6,
            "Entropie of Von Neumann site 6v6":  S_01,
            # "Distance Gilbert 5min": dist,
        }]
        with open(f'chain{v}.json', 'w') as f:
            json.dump(data_to_dump, f)