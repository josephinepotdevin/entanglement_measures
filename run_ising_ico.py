
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt
import time
import json
import sys

from ising_utils import (
    index_to_bitstring,
    ket_string_from_state,
    log_negativity,
    log_negativity_pure_bipartition,
    product_state_z,
    # product_half_up_half_down,
    # random_half_up_half_down
)

def ising_sparse_ico(N, edges, J=1.0, h=1.0):
    """
    Transverse-Field Ising Model (TFIM) on Icosahedron graph. 
    Warning: operators x and z inverted from standard definition.
    H = -h sum Z_i - J sum X_i X_j
    Hardcoded for N=12 vertices.
    """
    if N != 12:
        print(f"Warning: Icosahedron graph is defined for N=12. You asked for N={N}.")
        # We proceed but we only use the edges defined for 0..11.
        # If N < 12, this will crash. If N > 12, extra spins are isolated (only h term).

    dim = 1 << N
    rows, cols, data = [], [], []

    # --- Diagonal term: -h sum Z_i ---
    for s in range(dim):
        e = 0.0
        for i in range(N):
            zi = 1.0 if ((s >> i) & 1) == 0 else -1.0
            e += -h * zi
        rows.append(s)
        cols.append(s)
        data.append(e)

    # --- Off-diagonal term: J sum X_i X_j ---
    # In Z-basis: X flips the bit.
    for s in range(dim):
        for (i,j) in edges:
            sp = s ^ (1 << i) ^ (1 << j)  # flip both spins i and j
            rows.append(sp)
            cols.append(s)
            data.append(J)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H

def main():
    parser = argparse.ArgumentParser(description='Simulate Transverse Field Ising Model on Icosahedron')
    parser.add_argument('--N', type=int, default=12, help='Number of spins (should be 12)')
    parser.add_argument('--J', type=float, default=1.0, help='Ising coupling J')
    parser.add_argument('--h', type=float, default=3.0, help='Transverse field h') # Default from file
    parser.add_argument('--tfin', type=float, default=1.4, help='Final time')
    parser.add_argument('--tsteps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--output', type=str, default='ising_ico_results', help='Output filename prefix')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()

    print(f"Running Ising Icosahedron Simulation with N={args.N}, J={args.J}, h={args.h}, tfin={args.tfin}")

    if args.N != 12:
        print("Warning: Edges hardcoded for 12 spins.")

    start_time = time.perf_counter()

    # Edges of an Icosahedron (12 vertices, 30 edges)
    # Vertex labeling 0..11
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
        (10,11)
    ]

    # Hamiltonian
    # Note: ising_ico.py implements H = -h Z - J XX
    H = ising_sparse_ico(args.N, edges, args.J, args.h)

    psi0 = product_state_z(args.N, down_sites=[0,2,6,10,11])
    # psi0 = np.zeros(dim, dtype=np.complex128) #for chain
    # psi0[0] = 1.0    # |000...0>

    # psi0 = product_half_up_half_down(N)

    t_values = np.linspace(0, args.tfin, args.tsteps)
    
    # Storage
    results = {
        "neg_01": [],      # site 3 vs 9 (indices 3 and 9 are kept)
        "neg_01_2": [],    # site 3,8 vs 9 (indices 3,8 and 9 kept, partition 3,8 | 9)
        "neg_01_23": [],   # site 2,3,8,9 kept? Wait. 
        "neg_3v3": [],
        "neg_6v6": []
    }

    # Time evolution
    for idx, t in enumerate(t_values):
        if idx % 10 == 0:
            print(f"Time step {idx}/{args.tsteps} (t={t:.2f})")
        
        psi_t = expm_multiply(-1j * H * t, psi0)
        psi_t = psi_t / np.linalg.norm(psi_t)
        dm_psi_t = DensityMatrix(psi_t)

        # Measurements
        # 1. RDM of site 3 and 9. (Trace out everything else)
        # Original: partial_trace(..., [0,1,2,4,5,6,7,8,10,11]) -> keeps 3, 9
        # Renaming key to be more clear in JSON but keeping variable names for sanity
        rdm_01 = partial_trace(dm_psi_t, [0,1,2,4,5,6,7,8,10,11]) 
        results["neg_01"].append(log_negativity(rdm_01, [0]))

        # 2. RDM of ...
        # Original: RDM_t_012 = partial_trace(..., [0,1,2,4,5,6,7,10,11]) 
        # Missing: 3, 8, 9. (3 qubits)
        # neg_t_01_2 = log_negativity(RDM_t_012, [0,1])
        # Transpose qubits 0 and 1 of RDM. (Which are 3 and 8 in ascending order?)
        # partial_trace preserves order? Yes.
        # Original indices: 0,1,2...11.
        # Removing 0,1,2,4,5,6,7,10,11.
        # Remaining: 3, 8, 9.
        # So qubit 0 of RDM is old 3, qubit 1 is old 8, qubit 2 is old 9.
        # Transpose [0,1] => Transpose 3 and 8.
        # Partition: {3,8} vs {9}.
        rdm_012 = partial_trace(dm_psi_t, [0,1,2,4,5,6,7,10,11])
        results["neg_01_2"].append(log_negativity(rdm_012, [0,1]))
        
        # 3.
        # Original: RDM_t_0123 = partial_trace(..., [0,1,4,5,6,7,10,11])
        # Missing: 2, 3, 8, 9. (4 qubits)
        # neg_t_01_23 = log_negativity(RDM_t_0123, [0,1])
        # Qubits of RDM: 2, 3, 8, 9.
        # Transpose 0,1 => old 2, 3.
        # Partition {2,3} vs {8,9}.
        rdm_0123 = partial_trace(dm_psi_t, [0,1,4,5,6,7,10,11])
        results["neg_01_23"].append(log_negativity(rdm_0123, [0,1]))
        
        # 4.
        # Original: RDM_t_3v3 = partial_trace(..., [0,1,4,5,6,10])
        # Missing: 2, 3, 7, 8, 9, 11. (6 qubits)
        # neg_t_3v3 = log_negativity(RDM_t_3v3, [0,1,2])
        # Transpose 0,1,2 => old 2,3,7? 
        # (Assuming sorted: 2,3,7,8,9,11)
        # Partition {2,3,7} vs {8,9,11}.
        rdm_3v3 = partial_trace(dm_psi_t, [0,1,4,5,6,10])
        results["neg_3v3"].append(log_negativity(rdm_3v3, [0,1,2]))
        
        # 5.
        # neg_t_6v6 = log_negativity_pure_bipartition(psi_t, NA=6)
        results["neg_6v6"].append(log_negativity_pure_bipartition(psi_t, NA=6))

    end_time = time.perf_counter()
    duration = (end_time - start_time) / 60
    print(f"Computation time: {duration:.2f} min")

    # Data dump
    data_to_dump = {
        "Info": f"Ico N={args.N} J={args.J} h={args.h}",
        "Computation time [min]": duration,
        "Ising J": args.J,
        "Ising h": args.h,
        "Edges": edges,
        "Time": t_values.tolist(),
        "Log Neg site 3 vs 9": results["neg_01"],
        "Log Neg site 3,8 vs 9": results["neg_01_2"],
        "Log Neg site 2,3 vs 8,9": results["neg_01_23"],
        "Log Neg site 2,3,7 vs 8,9,11": results["neg_3v3"],
        "Log Neg 6v6": results["neg_6v6"]
    }
    
    with open(f'{args.output}.json', 'w') as f:
        json.dump(data_to_dump, f, indent=4)
        print(f"Data saved to {args.output}.json")

    # Plotting
    if not args.no_plot:
        colors = ['aquamarine','coral', 'springgreen','purple', 'dimgrey','gold']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(t_values, results["neg_01"], label=r'$\epsilon_{0|1} (3|9)$', color=colors[0], marker='x', linestyle='-', linewidth=3)
        ax.plot(t_values, results["neg_01_2"], label=r'$\epsilon_{01|2} (3,8|9)$', color=colors[1], marker='x', linestyle='-', linewidth=2)
        ax.plot(t_values, results["neg_01_23"], label=r'$\epsilon_{01|23} (2,3|8,9)$', color=colors[2], marker='x', linestyle='-', linewidth=2)
        ax.plot(t_values, results["neg_3v3"], label=r'$\epsilon_{3v3}$', color=colors[3], marker='x', linestyle='-', linewidth=2)
        ax.plot(t_values, results["neg_6v6"], label=r'$\epsilon_{6v6}$', color=colors[4], marker='x', linestyle='-', linewidth=2)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Logarithmic negativity $\epsilon$')
        ax.set_title(f'Ising Icosahedron h={args.h}, J={args.J}')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        fig.savefig(f'{args.output}.png', dpi=300, bbox_inches='tight')

        plt.show()

        print(f"Plot saved to {args.output}.png")

if __name__ == '__main__':
    main()
