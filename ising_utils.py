import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace

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
