import numpy as np
import torch

from scipy.stats import multivariate_normal


def get_cov_in_p_dims(i=1, p=3, tensor=False):
    if i != 1:
        raise ValueError("i must be 1")
    if i == 1:
        c1 = np.diag([5] + [1] * (p - 1))
        c2 = np.diag([1] * (p - 1) + [5])
    if tensor:
        return torch.tensor(c1, dtype=torch.float32), torch.tensor(c2, dtype=torch.float32)
    else: 
        return c1, c2
    

# def random_cov(p):
#     # eigenvalues = np.linspace(0.5, 1.5, p)  # Eigenvalues between 0.5 and 1.5
#     max_eig = np.random.uniform(0.5, 1.5)
#     eigs = np.random.uniform(0.5, max_eig, p-1)
#     eigs = np.append(eigs, max_eig)
    
#     # Step 4: Perform eigen decomposition on the symmetrized matrix
#     Q, _ = np.linalg.qr(np.random.rand(p, p))  # Random orthogonal matrix
#     uniform_cov_matrix = Q @ np.diag(eigs) @ Q.T  # Reconstruct covariance matrix
    
#     return uniform_cov_matrix


# Generate a random covariance matrix of size p with uniform eigval_max/sum(eigvals)
def random_cov(p, a=0.1, b=0.9):
    # Sample ratio r uniformly
    r = np.random.uniform(a, b)
    
    # Generate random eigenvalues
    eigvals = np.random.uniform(0.1, 1, p-1)  # Generate base eigenvalues
    eigval_max = r / (1 - r) * np.sum(eigvals)  # Calculate max eigenvalue
    eigvals = np.append(eigvals, eigval_max)  # Add max eigenvalue
    np.random.shuffle(eigvals)
    
    # Generate random orthonormal matrix Q
    random_matrix = np.random.randn(p, p)
    Q, _ = np.linalg.qr(random_matrix)
    
    # Construct covariance matrix
    cov_matrix = Q @ np.diag(eigvals) @ Q.T
    
    # # Ensure symmetry and positive semi-definiteness
    # cov_matrix = (cov_matrix + cov_matrix.T) / 2
    return cov_matrix


def get_random_cov(p=3, e=2, tensor=False):
    covs = [random_cov(p) for _ in range(e)]
    # covs = [cov @ cov.T for cov in covs]
    if tensor:
        return [torch.tensor(cov, dtype=torch.float32) for cov in covs]
    else:
        return covs


def get_ith_covs_rank3(i=1, tensor=False):
    if i > 7 or i < 1:
        raise ValueError("i must be between 1 and 7")
    rank3_covs = {
        # Different correlation structure
        1: [torch.tensor([[1, 0.5, 0], [0.5, 1, 0.1], [0, 0.1, 1]], dtype=torch.float32),
            torch.tensor([[1, -0.5, 0], [-0.5, 1, 0.1], [0, 0.1, 1]], dtype=torch.float32)],
        # Different correlation structure
        2: [torch.tensor([[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]], dtype=torch.float32),
            torch.tensor([[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]], dtype=torch.float32)],
        # Same ones
        3: [torch.tensor([[1, 0.5, 0.3], [0.5, 1, 0.1], [0.3, 0.1, 1]], dtype=torch.float32),
            torch.tensor([[1, 0.5, 0.1], [0.5, 1, 0.1], [0.1, 0.1, 1]], dtype=torch.float32)],
        # Opposite correlations
        4: [torch.tensor([[1, -0.5, -0.3], [-0.5, 1, -0.1], [-0.3, -0.1, 1]], dtype=torch.float32),
            torch.tensor([[1, 0.5, 0.3], [0.5, 1, 0.1], [0.3, 0.1, 1]], dtype=torch.float32)],
        # Orthogonal first eigenvectors
        5: [torch.tensor([[5, 0, 0], [0, 4, 0], [0, 0, 1]], dtype=torch.float32),
            torch.tensor([[1, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=torch.float32)],
        # Illustrating unnormalized PCA vs. reconstruction
        6: [torch.tensor([[5, 0, 0], [0, 4, 0], [0, 0, 1]], dtype=torch.float32),
            torch.tensor([[0.2, 0, 0], [0, 0.5, 0], [0, 0, 0.8]], dtype=torch.float32)],
        # First one with almost equal eigenvalues, second strongly third
        7: [torch.tensor([[0.35, 0, 0], [0, 0.33, 0], [0, 0, 0.32]], dtype=torch.float32),
            torch.tensor([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.9]], dtype=torch.float32)]
    }
    c1, c2 = rank3_covs[i]
    if tensor:
        return c1, c2
    else:
        return c1.numpy(), c2.numpy()
    

def get_ith_covs(i=1, p=3, e=2, random=False, tensor=True):
    if random:
        covs = get_random_cov(p=p, e=e, tensor=tensor)
    elif p == 3:
        covs = get_ith_covs_rank3(i=i, tensor=tensor)
    elif p > 3:
        covs = get_cov_in_p_dims(i=i, p=p, tensor=True)
    return {'covs': covs, 'norm_csts': [np.trace(cov) for cov in covs]}


def get_X_from_cov_params(params, p=3, n=1000, tensor=True):
    # Generate multivariate normal samples
    Xs = [multivariate_normal.rvs(mean=np.zeros(p), cov=cov, size=n) 
          for cov in params['covs']]
    
    # Center the data (scale with mean zero, but without standard deviation scaling)
    Xs = [X - np.mean(X, axis=0) for X in Xs]

    # Compute covariance matrices
    covs = [X.T @ X for X in Xs]

    if tensor:
        return {'Xs': [torch.tensor(X, dtype=torch.float32) for X in Xs], 
                'covs': [torch.tensor(cov, dtype=torch.float32) for cov in covs],
                'norm_csts': [np.trace(cov) for cov in covs]}
    else:
        return {'Xs': Xs, 
                'covs': covs, 
                'norm_csts': [np.trace(cov) for cov in covs]}


def get_ith_X_and_covs(i=1, p=3,  e=2, random=False, n=1000, tensor=True):
    # Assuming get_ith_covs_R is defined in Python and returns a list of covariance matrices
    params = get_ith_covs(i=i, p=p, random=random, e=e, tensor=False)
    return get_X_from_cov_params(params, p=p, n=n, tensor=tensor)
    
    