import numpy as np
import cvxpy as cp
from itertools import product

def partial_transpose(rho, dims, sys):
    """
    Calculates the partial transpose of a density matrix rho.
    
    Args:
        rho: Density matrix (numpy array or cvxpy Variable/Expression).
        dims: List of dimensions of the subsystems [d1, d2, ...].
        sys: List of subsystems (0-indexed) to transpose.
        
    Returns:
        The partially transposed matrix.
    """
    # Verify dimensions match rho size
    if np.prod(dims) != rho.shape[0]:
        raise ValueError("Dimensions do not match size of rho.")
    
    num_subsystems = len(dims)
    
    # Reshape rho into a tensor with 2*num_subsystems indices:
    # (row_sys_0, row_sys_1, ..., row_sys_n-1, col_sys_0, col_sys_1, ..., col_sys_n-1)
    shape = dims + dims
    
    # If rho is a cvxpy object, we need to handle reshaping carefully if it supports it,
    # but cvxpy generic reshaping can be tricky with complex indices. 
    # Usually easier to define permutation on indices.
    # However, for simply permuting axes, reshape -> transpose -> reshape works for both numpy and recent cvxpy.
    
    # Subsystem indices in the tensor:
    # Row indices: 0 to num_subsystems-1
    # Col indices: num_subsystems to 2*num_subsystems-1
    
    perm = list(range(2 * num_subsystems))
    
    # For each subsystem to transpose, swap its row and col index
    for s in sys:
        row_idx = s
        col_idx = s + num_subsystems
        perm[row_idx], perm[col_idx] = perm[col_idx], perm[row_idx]
        
    if isinstance(rho, (cp.Variable, cp.expressions.expression.Expression, cp.atoms.affine.reshape.reshape)):
        # CVXPY implementation using vectorization to avoid dimension > 2 warnings
        # 1. Create a dummy index matrix 0..N^2-1
        # 2. Reshape to tensor, transpose indices, flatten to get permutation
        
        total_dim = np.prod(dims)
        dummy_indices = np.arange(total_dim**2).reshape(shape, order='F')
        dummy_pt = dummy_indices.transpose(perm)
        perm_indices = dummy_pt.flatten(order='F')
        
        # Flatten input rho, permute, then reshape back
        # cp.vec(rho) flattens in column-major order ('F')
        rho_vec = cp.vec(rho, order='F')
        rho_pt_vec = rho_vec[perm_indices]
        
        return cp.reshape(rho_pt_vec, rho.shape, order='F')
        
    else:
        # Numpy implementation
        rho_tensor = rho.reshape(shape, order='F')
        rho_pt = rho_tensor.transpose(perm)
        return rho_pt.reshape(rho.shape, order='F')

def get_partitions(n):
    """
    Generates all inequivalent bipartitions M|complement(M) for n systems.
    Returns a list of binary vectors (lists of 0s and 1s) representing M.
    """
    partitions = []
    # Loop from 1 to 2^(n-1) - 1
    # We only go up to half to avoid doing both M|M^c and M^c|M
    for i in range(1, 2**(n - 1)):
        # Convert to binary string, pad with zeros
        bin_str = format(i, f'0{n}b')
        # Convert to list of integers
        partition = [int(b) for b in bin_str]
        partitions.append(partition)
    return partitions

def fdecwit(rho, dims=None, solver=None, verbose=False):
    """
    Returns the minimal expectation value of rho with respect to all fully
    decomposable witnesses and the witness itself.
    
    Corresponds to fdecwit.m
    """
    if dims is None:
        # Assume qubits
        n_qubits = int(np.log2(rho.shape[0]))
        dims = [2] * n_qubits
    
    if np.prod(dims) != rho.shape[0]:
        raise ValueError("Dimensions do not match rho size")
            
    n_sys = len(dims)
    op_dims = rho.shape
    
    # Define SDP variable for Witness W
    W = cp.Variable(op_dims, hermitian=True)
    
    constraints = [cp.trace(W) == 1]
    
    parts = get_partitions(n_sys)
    
    # Store auxiliary variables to keep them in scope/graph
    Ps = []
    
    for partition in parts:
        # M is list of 0s and 1s. 1 indicates belonging to partition set M.
        # We need to partial transpose W - P w.r.t M.
        # The MATLAB logic:
        # P{m} >= 0
        # (W - P{m})^TM >= 0
        
        P = cp.Variable(op_dims, hermitian=True)
        Ps.append(P)
        
        # Identify subsystems to transpose (indices where partition is 1)
        # In MATLAB pt.m: "returns partial transpose ... w.r.t particles indicated by ONE in binary vector"
        sys_to_transpose = [i for i, x in enumerate(partition) if x == 1]
        
        constraints.append(P >> 0)
        constraints.append(partial_transpose(W - P, dims, sys_to_transpose) >> 0)
        
    # Objective: Minimize trace(rho * W)
    # Note: rho is constant here. trace(rho * W) is real if rho, W hermitian.
    obj = cp.Minimize(cp.real(cp.trace(rho @ W)))
    
    prob = cp.Problem(obj, constraints)
    
    # Solve
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        print(f"Solver failed: {e}")
        return None, None

    min_val = prob.value
    witness = W.value
    
    # Check "negative" condition from MATLAB ( < -1e-12)
    if min_val is None or min_val >= -1e-12:
        witness = np.zeros(op_dims)
        
    return min_val, witness

def fpptwit(rho, dims=None, solver=None, verbose=False):
    """
    Returns the minimal expectation value of rho with respect to all fully
    PPT witnesses.
    
    Corresponds to fpptwit.m
    """
    if dims is None:
        n_qubits = int(np.log2(rho.shape[0]))
        dims = [2] * n_qubits
        
    n_sys = len(dims)
    op_dims = rho.shape
    
    W = cp.Variable(op_dims, hermitian=True)
    constraints = [cp.trace(W) == 1]
    
    parts = get_partitions(n_sys)
    
    for partition in parts:
        # Constraint: W^TM >= 0
        sys_to_transpose = [i for i, x in enumerate(partition) if x == 1]
        constraints.append(partial_transpose(W, dims, sys_to_transpose) >> 0)
        
    obj = cp.Minimize(cp.real(cp.trace(rho @ W)))
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        print(f"Solver failed: {e}")
        return None, None
        
    min_val = prob.value
    witness = W.value
    
    if min_val is None or min_val >= -1e-12:
        witness = np.zeros(op_dims)
        
    return min_val, witness

def entmon(rho, dims=None, solver=None, verbose=False):
    """
    Returns the entanglement monotone.
    Corresponds to entmon.m
    """
    if dims is None:
        n_qubits = int(np.log2(rho.shape[0]))
        dims = [2] * n_qubits
        
    n_sys = len(dims)
    op_dims = rho.shape
    
    W = cp.Variable(op_dims, hermitian=True)
    
    # No trace constraint on W in entmon.m?
    # Checking entmon.m code:
    # "W=sdpvar..."
    # No "trace(W)==1" constraint visible in the file snippet I read.
    # Wait, let me check the file content of entmon.m again in the history.
    # It says:
    # P{m}=sdpvar...
    # P{m} >= 0
    # Q = (W - P{m})^TM >= 0
    # I - P{m} >= 0
    # I - Q >= 0
    # Minimize trace(rho * W)
    # Return -min_val (so max trace(rho*W)?)
    # Wait, entmon returns "N = -double(trace(rho*W))".
    # And minimizes trace(rho*W).
    # So it calculates the negative of the minimum expectation value.
    
    parts = get_partitions(n_sys)
    constraints = []
    Ps = []
    
    eye_mat = np.eye(op_dims[0])
    
    for partition in parts:
        P = cp.Variable(op_dims, hermitian=True)
        Ps.append(P)
        sys_to_transpose = [i for i, x in enumerate(partition) if x == 1]
        
        Q = partial_transpose(W - P, dims, sys_to_transpose)
        
        constraints.append(P >> 0)
        constraints.append(Q >> 0)
        constraints.append(eye_mat - P >> 0)
        constraints.append(eye_mat - Q >> 0)
        
    obj = cp.Minimize(cp.real(cp.trace(rho @ W)))
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        print(f"Solver failed: {e}")
        return None

    return -prob.value
