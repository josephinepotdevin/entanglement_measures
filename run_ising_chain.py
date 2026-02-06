import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt
import time
import json
import sys
import os

from ising_utils import (
    index_to_bitstring,
    ket_string_from_state,
    log_negativity,
    log_negativity_pure_bipartition,
    entropy_von_neumann,
    product_state_z,
    product_half_up_half_down,
    random_half_up_half_down
)

from entanglement_utils import (
    fdecwit,
    fpptwit,
    entmon,
    partial_transpose,
)

def ising_sparse_chain(N, J=1.0, h=1.0, periodic=True):
    """
    Transverse-Field Ising Model (TFIM) in the Z-basis:
        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    """
    dim = 1 << N
    rows, cols, data = [], [], []

    # --- Diagonal term: -J sum Z_i Z_{i+1} ---
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
    for s in range(dim):
        for i in range(N):
            sp = s ^ (1 << i)  # flip spin i
            rows.append(sp)
            cols.append(s)
            data.append(-h)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    return H

def main():
    parser = argparse.ArgumentParser(description='Simulate Transverse Field Ising Model on a Chain')
    parser.add_argument('--N', type=int, default=12, help='Number of spins')
    parser.add_argument('--J', type=float, default=1.0, help='Ising coupling J')
    parser.add_argument('--h', type=float, default=1.0, help='Transverse field h')
    parser.add_argument('--tfin', type=float, default=10.0, help='Final time')
    parser.add_argument('--tsteps', type=int, default=150, help='Number of time steps')
    parser.add_argument('--periodic', action='store_true', help='Use periodic boundary conditions')
    parser.add_argument('--output', type=str, default='ising_chain_results', help='Output filename prefix')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()

    print(f"Running Ising Chain Simulation with N={args.N}, J={args.J}, h={args.h}, tfin={args.tfin}, steps={args.tsteps}, periodic={args.periodic}")

    start_time = time.perf_counter()

    # Hamiltonian
    H = ising_sparse_chain(args.N, args.J, args.h, args.periodic)
    dim = 1 << args.N
    dimensions = np.full(args.N, 2)
    Nmin = 4
    # Initial State

    psi0 = np.zeros(dim, dtype=np.complex128) #for chain
    psi0[0] = 1.0    # |000...0>
    
    # psi0 = product_half_up_half_down(N)

    #down_sites = list(range(args.N))
    #if args.N > 0:
    #   down_sites.remove(args.N // 2) # Example setup: all down except middle
    #psi0 = product_state_z(args.N, down_sites)

    t_values = np.linspace(0, args.tfin, args.tsteps)
    
    # Storage
    results = {
        "neg_01": [], "neg_02": [], "neg_03": [], "neg_04": [], "neg_05": [], "neg_06": [],
        "neg_012": [], "neg_0_123": [], "neg_01_23": [], "neg_3v3": [], "neg_6v6": [],
        "S_01": [], "GMN": []
    }

    # Time evolution
    for idx, t in enumerate(t_values):
        if idx % 10 == 0:
            print(f"Time step {idx}/{args.tsteps} (t={t:.2f})")
        
        psi_t = expm_multiply(-1j * H * t, psi0)
        psi_t = psi_t / np.linalg.norm(psi_t)
        
        dm_psi_t = DensityMatrix(psi_t)

        # Measurements 
        if args.N >= Nmin:
            # negativity
            # subsystems 0 vs 1
            results["neg_01"].append(log_negativity(partial_trace(dm_psi_t, list(range(2, args.N)))))
            
            # 0 vs 2 -> keep 0, 2
            # results["neg_02"].append(log_negativity(partial_trace(dm_psi_t, [1] + list(range(3, args.N)))))

            # 0 vs 3
            # results["neg_03"].append(log_negativity(partial_trace(dm_psi_t, [1, 2] + list(range(4, args.N)))))

            # 0 vs 4
            # results["neg_04"].append(log_negativity(partial_trace(dm_psi_t, [1, 2, 3] + list(range(5, args.N)))))
            
            # 0 vs 5
            # results["neg_05"].append(log_negativity(partial_trace(dm_psi_t, [1, 2, 3, 4] + list(range(6, args.N)))))
            
            # 0 vs 6
            # results["neg_06"].append(log_negativity(partial_trace(dm_psi_t, [1, 2, 3, 4, 5] + list(range(7, args.N)))))
            
            # 0 vs 12 (keep 0, 1, 2)
            results["neg_012"].append(log_negativity(partial_trace(dm_psi_t, list(range(3, args.N)))))

            # 0 vs 123 (keep 0, 1, 2, 3)
            rdm_0123 = partial_trace(dm_psi_t, list(range(4, args.N)))
            results["neg_0_123"].append(log_negativity(rdm_0123))
            
            # 01 vs 23
            results["neg_01_23"].append(log_negativity(rdm_0123, [0, 1]))
            
            # 3v3 (0,1,2 vs 3,4,5) -> keep 0..5
            #rdm_3v3 = partial_trace(dm_psi_t, list(range(6, args.N)))
            #results["neg_3v3"].append(log_negativity(rdm_3v3, [0, 1, 2]))
            
            # 6v6 pure partition
            #results["neg_6v6"].append(log_negativity_pure_bipartition(psi_t, NA=6))
            
            # entropy
            #results["S_01"].append(entropy_von_neumann(dm_psi_t, [0, 1, 2, 3, 4, 5]))
            
            # GMN
            # gmn = entmon(np.outer(psi_t, psi_t))
            # results["GMN"].append(gmn)
            # print(gmn)

        else    :
             # Basic fallback for smaller N
             if args.N >= 2:
                 # Just 0 vs 1
                 results["neg_01"].append(log_negativity(partial_trace(dm_psi_t, list(range(2, args.N)))))

    end_time = time.perf_counter()
    duration = (end_time - start_time) / 60
    print(f"Computation time: {duration:.2f} min")

    # Data Dump
    data_to_dump = {
        "Info": f"Chain N={args.N} J={args.J} h={args.h}",
        "N": args.N,
        "J": args.J,
        "h": args.h,
        "periodic": args.periodic,
        "Computation time [min]": duration,
        "Initial state": ket_string_from_state(psi0, args.N),
        "Time": t_values.tolist(),
        **results
    }
    
    with open(f'{args.output}.json', 'w') as f:
        json.dump(data_to_dump, f, indent=4)
        print(f"Data saved to {args.output}.json")

    # Plotting
    if not args.no_plot and args.N >= Nmin:
        colors = ['aquamarine', 'coral', 'springgreen','purple', 'dimgrey','gold']
        
        # Fig 1
        # fig, ax = plt.subplots(figsize=(8, 5))
        # ax.plot(t_values, results["neg_01"], label=r'$\epsilon_{0|1}$', color=colors[0])
        # ax.plot(t_values, results["neg_02"], label=r'$\epsilon_{0|2}$', color=colors[1])
        # ax.plot(t_values, results["neg_03"], label=r'$\epsilon_{0|3}$', color=colors[2])
        # ax.plot(t_values, results["neg_04"], label=r'$\epsilon_{0|4}$', color=colors[3])
        # ax.plot(t_values, results["neg_05"], label=r'$\epsilon_{0|5}$', color=colors[4])
        # ax.plot(t_values, results["neg_06"], label=r'$\epsilon_{0|6}$', color=colors[5])
        # ax.set_xlabel('Time [s]')
        # ax.set_ylabel(r'Logarithmic negativity $\epsilon$')
        # ax.set_title(f'Ising Chain N={args.N} h={args.h}, J={args.J}')
        # ax.legend()
        # ax.grid(True, linestyle=':', alpha=0.7)
        # fig.savefig(f'{args.output}_plot1.png', dpi=300, bbox_inches='tight')
        
        # Fig 2
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(t_values, results["neg_01"], label=r'$\epsilon_{0|1}$', color=colors[0])
        ax2.plot(t_values, results["neg_012"], label=r'$\epsilon_{0|12}$', color=colors[1])
        ax2.plot(t_values, results["neg_0_123"], label=r'$\epsilon_{0|123}$', color=colors[2])
        ax2.plot(t_values, results["neg_01_23"], label=r'$\epsilon_{01|23}$', color=colors[3])
        # ax2.plot(t_values, results["neg_3v3"], label=r'$\epsilon_{3v3}$', color=colors[4])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel(r'Logarithmic negativity $\epsilon$')
        ax2.set_title(f'Ising Chain N={args.N} h={args.h}, J={args.J}')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.7)
        fig2.savefig(f'{args.output}_plot2.png', dpi=300, bbox_inches='tight')

        # Fig 3
        # fig3, ax3 = plt.subplots(figsize=(8, 5))
        # ax3.plot(t_values, results["neg_6v6"], label=r'$\epsilon_{6v6}$', color=colors[0])
        # ax3.plot(t_values, results["S_01"], label=r'$S_{6v6}$', color=colors[1])
        # ax3.set_xlabel('Time [s]')
        # ax3.set_ylabel('Entanglement')
        # ax3.set_title(f'Ising Chain N={args.N} h={args.h}, J={args.J}')
        # ax3.legend()
        # ax3.grid(True, linestyle=':', alpha=0.7)
        # fig3.savefig(f'{args.output}_plot3.png', dpi=300, bbox_inches='tight')
        

        # Fig 4
        # fig4, ax4 = plt.subplots(figsize=(8, 5))
        # ax4.plot(t_values, results["neg_01"], label=r'$\epsilon_{0|1}$', color=colors[0])
        # ax4.plot(t_values, results["GMN"], label=r'$GMN$', color=colors[2])
        # ax4.set_xlabel('Time [s]')
        # ax4.set_ylabel('Entanglement')
        # ax4.set_title(f'Ising Chain N={args.N} h={args.h}, J={args.J}')
        # ax4.legend()
        # ax4.grid(True, linestyle=':', alpha=0.7)
        # fig4.savefig(f'{args.output}_plot4.png', dpi=300, bbox_inches='tight')

        plt.show()

        print(f"Plots saved to {args.output}_plot*.png")

if __name__ == '__main__':
    main()
