import numpy as np
import torch

# ----------------- Generate covairance matrices ----------------- #
def get_random_covs(p, rank, nenvs, 
                    a1=0.1, b1=0.9, a2=0.1, b2=0.9, 
                    full_cov=True):
    """
    Generate nenvs random covariance matrices of size pxp
    with rank `rank` shared component and rank `rank` env-specific component.
    Eigenvalues of shared and env-specific components are preserved exactly.

    Input
    p : the size of the covariance matrix
    rank : the rank of the covariance matrix
    nenvs : number of environments
    a1 : the lower bound for the eigenvalues of the shared component
    b1 : the upper bound for the eigenvalues of the shared component
    a2: the lower bound for the eigenvalues of the env-specific component
    b2 : the upper bound for the eigenvalues of the env-specific component
    full_cov : if True, return the full covariance matrix
                otherwise, return the eigenvalues and the orthogonal matrix
    """
    assert full_cov, "This function only supports full_cov=True for now."
    
    covs = []

    # 1. Generate shared eigenvalues
    eigvals_shared = np.random.uniform(a1, b1, rank)
    
    # 2. Generate env-specific eigenvalues
    eigvals_env = np.random.uniform(a2, b2, rank)
    
    for _ in range(nenvs):
        # 3. Shared random orthonormal basis
        Q0, _ = np.linalg.qr(np.random.randn(p, rank))
        
        # 4. Environment-specific orthonormal basis, orthogonal to Q0
        Qi = np.random.randn(p, rank)
        # Make orthogonal to Q0
        Qi -= Q0 @ (Q0.T @ Qi)
        # Orthonormalize
        Qi, _ = np.linalg.qr(Qi)
        
        # 5. Stack into single orthonormal basis
        Qtilde = np.concatenate([Q0, Qi], axis=1)  # p x 2*rank
        Lambda_tilde = np.diag(np.concatenate([eigvals_shared, eigvals_env]))
        
        # 6. Construct covariance
        C = Qtilde @ Lambda_tilde @ Qtilde.T
        covs.append(C)
        
    return [C / np.linalg.trace(C) for C in covs]


# ---------------- Helpers ----------------- #

def orthogonalize(v):
    """ Orthogonalize the columns of v using QR decomposition. """
    rank = v.shape[1]
    u, _, vh = torch.linalg.svd(v)
    v = u[:, :rank] @ vh
    return v


def reshape_v(v, X):
    """ Reshape v to have the correct number of rows based on X. """
    p = X.shape[1]
    return v.reshape(p, -1)

# ----------------- Objective functions ----------------- #


def f_pca(v, cov_matrix, norm_cst):
    v = reshape_v(v, cov_matrix)
    numerator = torch.trace(torch.mm(v.T, torch.mm(cov_matrix, v)))
    denominator = norm_cst
    return numerator / denominator


def f_reconstruction(v, cov_matrix, norm_cst, from_cov=False):
    if not from_cov:
        raise ValueError("from_cov=False not implemented yet")
    v = reshape_v(v, cov_matrix)
    # if from_cov:
    #     numerator = torch.trace(X1) - torch.trace(v.T @ X1 @ v)
    # else:
    #     numerator = torch.linalg.matrix_norm(X1 - X1 @ v @ v.T)**2 
    numerator = torch.trace(cov_matrix) - torch.trace(v.T @ cov_matrix @ v)
    denominator = norm_cst
    return numerator / denominator


def f_minpca(v, covs, norm_csts):
    v = reshape_v(v, covs[0])
    # check if v is orthogonal
    if not torch.allclose(torch.mm(v.T, v), torch.eye(v.shape[1]), atol=1e-5):
        v = orthogonalize(v)
    fs =[-f_pca(v, cov, norm_cst).unsqueeze(0).unsqueeze(1) 
         for cov, norm_cst in zip(covs, norm_csts)]
    fs_cat = torch.cat(fs, dim=1)  # fs_cat should now be of shape [1, N]
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0]


def f_minpca_pen(v, covs, norm_csts, c):
    if c < 0 or c > 1:
        raise ValueError("c must be between 0 and 1")
    
    avg_cov = torch.zeros_like(covs[0])
    for cov in covs:
        avg_cov += cov
    avg_cov /= len(covs)

    # Note: f_minpca takes -f_pca, so we need to negate the second term
    minpca_val = f_minpca(v, covs, norm_csts)
    pca_val = - f_pca(v, avg_cov, torch.trace(avg_cov))
    return c * minpca_val + (1-c) * pca_val


def f_minpca_memory(v, covs, norm_csts, prev_values):
    v = reshape_v(v, covs[0])
    v = orthogonalize(v)
    fs = [prev_values[i]-f_pca(v, covs[i], norm_csts[i]).unsqueeze(0).unsqueeze(1) 
          for i in range(len(covs))]
    fs_cat = torch.cat(fs, dim=1) 
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0]


def f_maxreconstruction(v, covs, norm_csts, from_cov=False):
    v = reshape_v(v, covs[0])
    if not torch.allclose(torch.mm(v.T, v), torch.eye(v.shape[1]), atol=1e-5):
        v = orthogonalize(v)
    fs =[f_reconstruction(v, X, norm_cst, from_cov).unsqueeze(0).unsqueeze(1) 
         for X, norm_cst in zip(covs, norm_csts)]
    fs_cat = torch.cat(fs, dim=1)
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0] 


# def f_fairpca(v, Xs, covs, norm_csts):
#     v = reshape_v(v, Xs[0])
#     k = v.shape[1]
#     opt_vs = [torch.tensor(np.linalg.eig(cov)[1][:, 0:k], dtype=torch.float32) 
#               for cov in covs]
#     fs = torch.tensor([f_reconstruction(v, Xs[i], norm_csts[i]) - 
#                        f_reconstruction(opt_vs[i], Xs[i], norm_csts[i]) 
#                        for i in range(len(Xs))], requires_grad=True)
#     return torch.max(fs)


def f_regret_variance(v, covs, norm_csts, opt_vals):
    v = reshape_v(v, covs[0])
    v = orthogonalize(v)
    fs = [opt_vals[i] - f_pca(v, covs[i], norm_csts[i]) 
          for i in range(len(covs))]
    fs = [f.unsqueeze(0).unsqueeze(1) for f in fs]
    fs_cat = torch.cat(fs, dim=1)
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0]


def f_regret_reconstruction(v, covs, norm_csts, opt_vals):
    v = reshape_v(v, covs[0])
    v = orthogonalize(v)
    fs = [f_reconstruction(v, covs[i], norm_csts[i], from_cov=True) - opt_vals[i] 
          for i in range(len(covs))]
    fs = [f.unsqueeze(0).unsqueeze(1) for f in fs]
    fs_cat = torch.cat(fs, dim=1)
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0]


# ----------------- Gradient functions ----------------- #

# # Example usage
# cov1 = np.array([[2.0, 0.5], [0.5, 1.0]])  # Sample covariance matrix 1
# cov2 = np.array([[1.0, 0.3], [0.3, 1.0]])  # Sample covariance matrix 2
# norm_csts = np.array([np.trace(cov1), np.trace(cov2)])  # Sample normalization constants
# v = np.array([1.0, 1.0])  # Initial vector

# # Compute gradient
# grad = gradient(v, cov1, cov2, norm_csts)
# print("Gradient:", grad)
def gradient(v, cov1, cov2, norm_csts):
    v = v.reshape(-1, 1)  # Ensure v is a column vector

    # Calculate f1 and f2
    f1 = -f_pca(v, cov1, norm_csts[0])  # Assume PCA is defined elsewhere
    f2 = -f_pca(v, cov2, norm_csts[1])  # Assume PCA is defined elsewhere

    if f1 > f2:
        # f1 is the active component, compute its gradient
        gradient_f1 = compute_gradient(v, cov1, norm_csts[0])
        return gradient_f1
    else:
        # f2 is the active component, compute its gradient
        gradient_f2 = compute_gradient(v, cov2, norm_csts[1])
        return gradient_f2


def compute_gradient(v, cov_matrix, norm_cst):
    # as below but torch
    norm_v_squared = torch.linalg.vector_norm(v, ord=2)**2
    term1 = 2 * cov_matrix @ v / norm_v_squared
    term2 = 2 * (v.T @ cov_matrix @ v) * v / norm_v_squared**2
    gradient = term1 - term2
    return -gradient.flatten() / norm_cst 


# ----------------- Helper functions ----------------- #

def project_out_vector(C, v):
    """
    Remove the component of covariance matrix C along vector v.

    Parameters:
        C (np.ndarray): pxp covariance matrix.
        v (np.ndarray): vector of length p.

    Returns:
        np.ndarray: New covariance matrix with v projected out.
    """
    # TODO: fix messy conversions between numpy and torch
    C = np.array(C)
    v = np.array(v.reshape(-1, 1))  # ensure column vector
    norm_sq = float(v.T @ v)
    u = v / np.sqrt(norm_sq) 
    P = np.eye(C.shape[0]) - u @ u.T
    C_proj = P @ C @ P

    return torch.tensor(C_proj, dtype=torch.float32)


# ----------------- Optimization functions ----------------- #

# TODO: could be neater
def get_solution_seq_minpca_memory(params, norm_csts, lr=0.1, betas=(0.9,0.99), 
                                   method='Adam', rank=1, prev_values=None, 
                                   n_iters=1000):
    if prev_values is None:
        prev_values = [0] * len(norm_csts)
 
    # Initialize the vector
    p = params['covs'][0].shape[1]
    v = torch.randn(p, requires_grad=True)
    
    # Choose optimizer
    optimizer_cls = torch.optim.Adam if method == 'Adam' else torch.optim.SGD
    optimizer = optimizer_cls([v], lr=lr, betas=betas if method == 'Adam' else None)

    # Optimizes
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = f_minpca_memory(v, params['covs'], norm_csts=norm_csts, 
                               prev_values=prev_values)
        loss.backward(retain_graph=i<999)
        optimizer.step()

    # Normalize and reshape the solution vector
    v = v / torch.norm(v)
    v = v.detach().clone().reshape((p, 1))
    
    if rank == 1:
        return v
    
    # Compute recursive step for higher ranks
    updated_values = [
        prev_values[i]-f_pca(v, params['covs'][i], params['norm_csts'][i]) 
        for i in range(len(prev_values))
    ]
    if params['Xs'] is not None:
        updated_Xs = [X - torch.mm(X, torch.mm(v, v.t())) for X in params['Xs']]
        updated_covs = [X.t() @ X for X in updated_Xs]
    else:
        updated_Xs = None
        updated_covs = [project_out_vector(cov, v) for cov in params['covs']]

    updated_params = {
        'Xs': updated_Xs, 
        'covs': updated_covs,
        'norm_csts': norm_csts
    }
    next_sol = get_solution_seq_minpca_memory(
        updated_params, norm_csts, lr=lr, betas=betas,
        method=method, rank=rank-1, prev_values=updated_values
    )
    
    # Concatenate and orthogonalize the solution vectors
    next_sol = torch.cat((v, next_sol), dim=1)
    u, _, vh = torch.linalg.svd(next_sol)
    v = u[:, :rank] @ vh
    return v
 
        
def get_solution(v_init, params, print_out=False, lr=0.1, betas=(0.9,0.99), 
                 c=0.9, method='Adam', function='minpca', rank=None, 
                 n_iters=1000):
    # TODO: remove v_init and rank from the function signature 
    # (only keep rank for e.g.,)
    if function == 'seq':
        seq_sol =  get_solution_seq_minpca_memory(
            params, norm_csts=params['norm_csts'], lr=lr, betas=betas, 
            method=method, rank=rank, n_iters=n_iters
        )
        return seq_sol
    rank = v_init.shape[1]
    # Initialize the vector
    v = v_init.clone().requires_grad_(True)

    # Select the function to optimize
    functions = {
        'minpca': f_minpca, 
        'minpca_pen': f_minpca_pen,
        'maxreconstruction': f_maxreconstruction, 
        # 'fairpca': f_fairpca,
        'pooled': f_minpca, #lambda v, cov, norm_cst: -f_pca(v, cov, norm_cst),
        'average': f_minpca, #lambda v, cov, norm_cst: -f_pca(v, cov, norm_cst),
        # 'joint_diag': None # Not implemented 
        'regret_variance': f_regret_variance,
        'regret_reconstruction': f_regret_reconstruction
    }
    f = functions[function]

    # Get arguments for the function
    covs = params['covs']
    norm_csts = params['norm_csts'] #[torch.trace(cov) for cov in covs]
    if function == 'minpca':
        args = {'covs': covs, 'norm_csts': norm_csts}
    elif function == 'maxreconstruction':
        args = {'covs': covs, 'norm_csts': norm_csts, 'from_cov': True}
    elif function == 'minpca_pen':
        args = {'covs': covs, 'norm_csts': norm_csts, 'c': c}
    # elif function == 'maxreconstruction':
    #     args = {'Xs': params['Xs'], 'norm_csts': norm_csts}
    # elif function == 'fairpca':
    #     args = params
    elif function == 'pooled':
        Xs = torch.cat(params['Xs'], dim=0)
        cov = torch.matmul(Xs.T, Xs) 
        args = {'covs': [cov], 'norm_csts': [torch.trace(cov)]}
    elif function == 'average':
        # Compute the average covariance matrix
        cov = torch.zeros_like(covs[0])
        for c in covs:
            cov += c
        cov /= len(covs)
        args = {'covs': [cov], 'norm_csts': [torch.trace(cov)]}
    elif function in ['regret_variance', 'regret_reconstruction']:
        args = {'covs': covs, 'norm_csts': norm_csts, 
                'opt_vals': params['opt_vals']}
    else:
        raise ValueError(f"Function {function} not recognized")

    # Select the optimizer
    if method == 'Adam':
        optimizer = torch.optim.Adam([v], lr=lr, betas=betas)
    else:
        optimizer = torch.optim.SGD([v], lr=lr)

    # Optimize
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = f(v, **args)
        loss.backward()
        
        if print_out & (i % 200 == 0) & (function == 'minpca'):
            if len(covs) > 2:
                print("Warning: more than 2 covariance matrices, `print_out` will not work")
            else:
                # Check gradient performs as expected
                grad_pytorch = v.grad.clone().tolist()
                v_tmp = v.clone().detach()
                grad_custom = gradient(v_tmp, covs[0], covs[1], norm_csts) 
                print(f"Iteration {i}")
                print(f"\tpytorch grad: [{', '.join(f'{g:.4f}' for g in grad_pytorch)}], ")
                print(f"\tcustom grad:  [{', '.join(f'{g:.4f}' for g in grad_custom)}]")
                print(f"\tloss = {loss.clone().flatten().item():.4f}")

        optimizer.step()

    # get svd of v & closest orthogonal matrix to v
    u, _, vh = torch.linalg.svd(v)
    v = u[:, :rank] @ vh
    return v #/ torch.linalg.vector_norm(v)
