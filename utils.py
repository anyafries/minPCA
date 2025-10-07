import autograd.numpy as anp
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import torch

from scipy.optimize import minimize

# ----------------- Objective functions ----------------- #


def f_pca(v, cov_matrix, norm_cst):
    p = cov_matrix.shape[0]
    v = v.reshape(p, -1)
    numerator = torch.trace(torch.mm(v.T, torch.mm(cov_matrix, v)))
    denominator = norm_cst
    # if v.shape[1] == 1:
    #     denominator *= torch.linalg.matrix_norm(v)**2
    return numerator / denominator


def f_reconstruction(v, X1, norm_cst):
    p = X1.shape[1]
    v = v.reshape(p, -1)
    numerator = torch.linalg.matrix_norm(X1 - X1 @ v @ v.T)**2 
    denominator = norm_cst #* torch.linalg.matrix_norm(v)**2
    return numerator / denominator


def f_minpca(v, covs, norm_csts):
    p = covs[0].shape[0]
    v = v.reshape(p, -1)
    # check if v is orthogonal
    if not torch.allclose(torch.mm(v.T, v), torch.eye(v.shape[1]), atol=1e-5):
        rank = v.shape[1]
        u, _, vh = torch.linalg.svd(v)
        v = u[:, :rank] @ vh
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
    p = covs[0].shape[0]
    v = v.reshape(p, -1)
    rank = v.shape[1]
    u, _, vh = torch.linalg.svd(v)
    v = u[:, :rank] @ vh
    fs = [prev_values[i]-f_pca(v, covs[i], norm_csts[i]).unsqueeze(0).unsqueeze(1) 
          for i in range(len(covs))]
    fs_cat = torch.cat(fs, dim=1) 
    return torch.max_pool1d(fs_cat, kernel_size=len(covs))[0,0]


def f_maxreconstruction(v, Xs, norm_csts):
    p = Xs[0].shape[1]
    v = v.reshape(p, -1)
    rank = v.shape[1]
    # get svd of v & closest orthogonal matrix to v
    u, _, vh = torch.linalg.svd(v)
    v = u[:, :rank] @ vh
    fs =[f_reconstruction(v, X, norm_cst).unsqueeze(0).unsqueeze(1) 
         for X, norm_cst in zip(Xs, norm_csts)]
    fs_cat = torch.cat(fs, dim=1)
    return torch.max_pool1d(fs_cat, kernel_size=len(Xs))[0,0] 


def f_fairpca(v, Xs, covs, norm_csts):
    p = Xs[0].shape[1]
    v = v.reshape(p, -1)
    k = v.shape[1]
    opt_vs = [torch.tensor(np.linalg.eig(cov)[1][:, 0:k], dtype=torch.float32) 
              for cov in covs]
    fs = torch.tensor([f_reconstruction(v, Xs[i], norm_csts[i]) - 
                       f_reconstruction(opt_vs[i], Xs[i], norm_csts[i]) 
                       for i in range(len(Xs))], requires_grad=True)
    return torch.max(fs)



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
        'fairpca': f_fairpca,
        'pooled': f_minpca, #lambda v, cov, norm_cst: -f_pca(v, cov, norm_cst),
        'average': f_minpca, #lambda v, cov, norm_cst: -f_pca(v, cov, norm_cst),
        # 'joint_diag': None # Not implemented yet
    }
    f = functions[function]

    # Get arguments for the function
    covs = params['covs']
    norm_csts = params['norm_csts'] #[torch.trace(cov) for cov in covs]
    if function == 'minpca':
        args = {'covs': covs, 'norm_csts': norm_csts}
    elif function == 'minpca_pen':
        args = {'covs': covs, 'norm_csts': norm_csts, 'c': c}
    elif function == 'maxreconstruction':
        args = {'Xs': params['Xs'], 'norm_csts': norm_csts}
    elif function == 'fairpca':
        args = params
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


PYOPTMAN_OPTIMIZERS = { 'SteepestDescent': pymanopt.optimizers.SteepestDescent(), 
                        'ConjugateGradient': pymanopt.optimizers.ConjugateGradient(),
                        'TrustRegions': pymanopt.optimizers.TrustRegions(),
                        'NelderMead': pymanopt.optimizers.NelderMead(),
                        'ParticleSwarm': pymanopt.optimizers.ParticleSwarm()}


def get_solution_pyoptman(cov1, cov2, dim=3, optimizer=pymanopt.optimizers.SteepestDescent()):
    cov1, cov2 = anp.array(cov1), anp.array(cov2)
    manifold = pymanopt.manifolds.Sphere(dim)

    @pymanopt.function.autograd(manifold)
    def cost(point):
        v1 = -point @ cov1 @ point / anp.trace(cov1)
        v2 = -point @ cov2 @ point / anp.trace(cov2)
        return anp.max(anp.array([v1, v2]))

    problem = pymanopt.Problem(manifold, cost)
    result = optimizer.run(problem)
    return result.point

