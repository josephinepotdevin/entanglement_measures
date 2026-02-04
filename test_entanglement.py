import numpy as np
import cvxpy as cp
from entanglement_utils import fdecwit, fpptwit, entmon, partial_transpose

def get_bell_state():
    # |Phi+> = (|00> + |11>)/sqrt(2)
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi)
    return rho

def get_separable_state():
    # Maximally mixed state for 2 qubits
    return np.eye(4) / 4

def test_partial_transpose():
    print("Testing Partial Transpose...")
    # Test on Bell state. PT of |Phi+><Phi+| is not positive.
    # PT(|Phi+><Phi+|) has eigenvalues 0.5, 0.5, 0.5, -0.5
    bell = get_bell_state()
    pt_bell = partial_transpose(bell, [2, 2], [1]) # Transpose 2nd qubit
    eigs = np.linalg.eigvalsh(pt_bell)
    print(f"Eigenvalues of PT(Bell): {eigs}")
    if np.min(eigs) < -1e-12:
        print("PASS: Partial transpose detected negative eigenvalue.")
    else:
        print("FAIL: Partial transpose should have negative eigenvalue.")

def test_fdecwit():
    print("\nTesting fdecwit...")
    bell = get_bell_state()
    val, wit = fdecwit(bell, [2, 2])
    print(f"Bell state value: {val}")
    if val < -1e-6:
        print("PASS: Bell state detected (value < 0).")
    else:
        print("FAIL: Bell state not detected.")
        
    sep = get_separable_state()
    val_sep, wit_sep = fdecwit(sep, [2, 2])
    print(f"Separable state value: {val_sep}")
    if val_sep > -1e-10:
        print("PASS: Separable state has value >= 0.")
    else:
        print("FAIL: Separable state shouldn't be detected.")

def test_fpptwit():
    print("\nTesting fpptwit...")
    bell = get_bell_state()
    val, wit = fpptwit(bell, [2, 2])
    print(f"Bell state value: {val}")
    if val < -1e-6:
        print("PASS: Bell state detected.")
    else:
        print("FAIL: Bell state not detected.")

def test_entmon():
    print("\nTesting entmon...")
    bell = get_bell_state()
    val = entmon(bell, [2, 2])
    print(f"Bell state entmon: {val}")
    if val > 0.0: # Should be positive for entangled? 
        # entmon returns -min_val. If min_val is negative, entmon is positive.
        print("PASS: Bell state has positive entmon.")
    else:
        print("FAIL: Bell state should have positive entmon.")

if __name__ == "__main__":
    try:
        test_partial_transpose()
        test_fdecwit()
        test_fpptwit()
        test_entmon()
    except ModuleNotFoundError:
        print("Error: cvxpy not properly installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
