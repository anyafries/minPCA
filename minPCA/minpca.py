import numpy as np
import torch
from minPCA.utils import get_solution, f_maxrcs, f_minpca, f_regret_variance

BASE_FUNCTIONS = ['minpca', 'maxrcs','maxregret']
DEVELOPMENT_FUNCTIONS = [
    'seq', 'pooled', 'average', #'fairpca', 
    'minpca_pen', 'regret_reconstruction'
]

def _to_torch(Xs):
    """Convert list of numpy arrays to list of torch tensors."""
    if isinstance(Xs[0], np.ndarray):
        return [torch.from_numpy(X).float() for X in Xs]
    return Xs


def get_errs_pca(v, params=None, params_pooled=None, from_cov=False):
    max_training = None 
    pooled = None
    if params is not None:
        arg2 = params['Xs'] if not from_cov else params['covs']
        max_training = f_maxrcs(
            v, arg2, params['norm_csts'], from_cov)
        max_training = max_training.detach().item()
    if params_pooled is not None:
        arg2 = params_pooled['Xs'] if not from_cov else params_pooled['covs']
        pooled = f_maxrcs(
            v, arg2, params_pooled['norm_csts'], from_cov)
        pooled = pooled.detach().item()
    return max_training, pooled


def get_vars_pca(v, params=None, params_pooled=None):
    max_training = None
    pooled = None
    if params is not None:
        max_training = -f_minpca(v, params['covs'], 
                                 params['norm_csts']).detach().item()
    if params_pooled is not None:
        pooled = -f_minpca(v, params_pooled['covs'], 
                           params_pooled['norm_csts']).detach().item()
    return max_training, pooled


def _get_score(v, params, function):
    if function == 'minpca':
        return -get_vars_pca(v, params)[0]
    elif function == 'maxrcs':
        return get_errs_pca(v, params, from_cov=True)[0]
    elif function == 'maxregret':
        return f_regret_variance(
            v, params['covs'], params['norm_csts'], params['opt_vals']
        ).detach().item()
    return None


def generate_params(Xs, pooled=False, norm=True, from_cov=False, center=True):
    Xs = _to_torch(Xs)
    # if from_cov is True, we assume that Xs are covariance matrices
    if from_cov:
        covs = Xs
        if pooled:
            covs = [torch.mean(torch.stack(covs), dim=0)]
        Xs_centred = None
    # otherwise, we assume that Xs are data matrices 
    # and we compute the covariance matrices
    else:
        if pooled:
            Xs = [torch.cat(Xs, dim=0)]
        if center:
            Xs = [X - X.mean(dim=0) for X in Xs]
        covs = [torch.cov(X.t(), correction=0)  for X in Xs]

    # if norm is True, we normalize the covariance matrices by their trace
    if norm: 
        norm_csts = [torch.trace(cov) for cov in covs]
    else: 
        norm_csts = [1] * len(covs)

    params = {'covs': covs, 'norm_csts': norm_csts}
    if not from_cov:
        params['Xs'] = Xs
    return params


class minPCA():
    """
    minPCA class: finds the (directions of) worst-case explained variance for 
    a set of environments

    Parameters
    ----------
    n_components : int
        Number of components to keep.
    function : str, default='minpca'
        The function to optimize. 
        Options are:
        - 'minpca': minimize the worst-case (normalized) explained variance
        - 'maxrcs': maximize the worst-case (normalized) reconstruction error
        - 'maxregret': maximize the worst-case (normalized) regret 
        Development options (not fully tested):
        - 'seq': sequential minPCA components
        - 'pooled': PCA on the pooled covariance matrix
        - 'average': PCA on the average covariance matrix
        - 'minpca_pen': penalized minPCA
        - 'regret_reconstruction': minimize the worst-case reconstruction regret
    norm : bool, default=True
        Whether to normalize the covariance matrices by their trace.
    
    Attributes
    ----------
    n_components_ : int
        Number of components to keep.
    function_ : str
        The function to optimize.
    norm_ : bool
        Whether to normalize the covariance matrices by their trace.
    K_ : int
        Number of environments.
    p_ : int
        Dimension of the data.
    Xs : list of torch.Tensor
        List of the data matrices.
    means_ : list of torch.Tensor
        List of the means of the data matrices.
    params_ : dict
        Dictionary of the parameters of the data matrices.
        This includes the centred matrices, the covariance matrices, and the 
        normalization constants.
    pooled_mean_ : torch.Tensor
        Mean of the pooled data.
    params_pooled_ : dict
        Parameters of the pooled data.
    v_ : torch.Tensor of shape (p_, n_components_)
        The solution of the optimization problem.

    PUBLIC: 
    maxerr_ : float
        The worst-case reconstruction error.
    pooled_err_ : float
        The pooled reconstruction error.
    minvar_ : float
        The worst-case explained variance.
    pooled_var_ : float
        The pooled explained variance.
    cumsum_minvar_ : list of float
        Cumulative minimum explained variance for each added component.
    cumsum_pooled_var_ : list of float
        Cumulative pooled explained variance for each added component.


    """
    def __init__(self, n_components, function='minpca', norm=True):
        # Initial checks
        assert isinstance(n_components, int), "`n_components` should be an integer."
        assert function in BASE_FUNCTIONS + DEVELOPMENT_FUNCTIONS, \
            f"function '{function}' not recognized."
        if function in DEVELOPMENT_FUNCTIONS:
            print(f"Warning: function '{function}' is in development mode.")
        assert isinstance(norm, bool), "`norm` should be a boolean."

        self.n_components_ = n_components
        self.function_ = function
        self.norm_ = norm

    def fit(self, Xs, n_iters=500, lr=0.1, betas=(0.9, 0.99), method='Adam',
            n_restarts=5, verbose=False, v0=None):
        assert isinstance(Xs, list)
        assert len(set(X.shape[1] for X in Xs)) == 1
        if self.function_ in DEVELOPMENT_FUNCTIONS and n_restarts > 1:
            print(f"Warning: n_restarts > 1 is not implemented for function '{self.function_}'.")
            print(f"Setting n_restarts to 1.")
            n_restarts = 1

        Xs = _to_torch(Xs)
        self.K_ = len(Xs)
        self.p_ = Xs[0].shape[1]
        self.Xs, self.means_, self.pooled_mean_ = None, None, None
        self.params_ = generate_params(Xs, norm=self.norm_, from_cov=True)
        self.params_pooled_ = generate_params(Xs, pooled=True, norm=self.norm_,
                                              from_cov=True)
        
        # initialize v0
        if v0 is None: 
            v0 = torch.randn(self.p_, self.n_components_)
        else:
            if isinstance(v0, np.ndarray):
                v0 = torch.from_numpy(v0).float()
            assert v0.shape == (self.p_, self.n_components_), \
                f"v0 should have shape ({self.p_}, {self.n_components_}), but got {v0.shape}."
            
        # for the regret: need the optimal variance/error per environemnt
        if self.function_ in ['maxregret', 'regret_reconstruction']:
            opt_vals = []
            for cov, norm_cst in zip(self.params_['covs'], self.params_['norm_csts']):
                eigvals, _ = torch.linalg.eig(cov / norm_cst)
                eigvals = torch.real(eigvals)
                eigvals, _ = torch.sort(eigvals, descending=True)
                if self.function_ == 'maxregret':
                    opt_vals.append(torch.sum(eigvals[:self.n_components_]).item())
                else: 
                    opt_vals.append(torch.sum(eigvals[self.n_components_:]).item())
            self.params_['opt_vals'] = opt_vals

        
        v0 = get_solution(v0, self.params_, function=self.function_, c=0.99, 
                               rank=self.n_components_, n_iters=n_iters,
                               lr=lr, betas=betas, method=method)
        best_v = v0
        if self.function_ in BASE_FUNCTIONS:
            best_score = _get_score(v0, self.params_, self.function_)

        # multiple restarts to avoid local minima, only for the main functions
        if n_restarts > 1:
            for _ in range(n_restarts - 1):
                v_try = torch.randn(self.p_, self.n_components_)
                v_try = get_solution(v_try, self.params_, function=self.function_, c=0.99,
                                  rank=self.n_components_, n_iters=n_iters,
                                  lr=lr, betas=betas, method=method)
                score_try = _get_score(v_try, self.params_, self.function_)
                if score_try < best_score:
                    if verbose: print(f"\timproving from {best_score:.3f} to {score_try:.3f}")
                    best_v, best_score = v_try, score_try

        # comparison to pool PCA
        if self.function_ in BASE_FUNCTIONS:
            cov_pooled = self.params_pooled_['covs'][0] / self.params_pooled_['norm_csts'][0]
            _, eigvecs_pooled = torch.linalg.eigh(cov_pooled)  # ascending order, real by construction
            v_pooled = eigvecs_pooled[:, -self.n_components_:].flip(dims=[1])
            pool_score = _get_score(v_pooled, self.params_, self.function_)
            if pool_score < best_score:
                if verbose: print(f"\tPool PCA improves score from {best_score:.3f} to {pool_score:.3f}. Using pool PCA solution.")
                best_v = v_pooled

        self.v_ = best_v

        self.maxerr_, self.pooled_err_ = get_errs_pca(
            self.v_, self.params_, params_pooled=self.params_pooled_,
            from_cov=True)
        
        self.minvar_, self.pooled_var_ = get_vars_pca(
            self.v_, self.params_, params_pooled=self.params_pooled_)
        
        self.cumsum_minvar_ = []
        self.cumsum_pooled_var_ = []
        for i in range(self.n_components_):
            var_i, pooled_var_i = get_vars_pca(self.v_[:, :i+1], self.params_, 
                                               self.params_pooled_)
            self.cumsum_minvar_.append(var_i)
            self.cumsum_pooled_var_.append(pooled_var_i)

        return self
    

    def components(self, ordered=False, lr=0.1, n_iters=1000):
        """
        Returns the components of the (min)PCA.
        If `self.function_` is 'minpca', the components are not ordered. 
        By setting `ordered=True`, the components are ordered by their
        explained variance:
        i.e., self.v_ is ordered to [v_1, v_2, ..., v_{n_components_}] 
                such that [v_1, ..., v_i] has the largest worst-case
                explained variance of a rank-i basis within span(v_)
        """
        if ordered and self.function_ == 'minpca':
            # the joint solution does not have a specific order, so we
            # need to find the optimal basis 
            v_remaining = self.v_.clone().detach()  
            v_ordered = []
            for i in range(self.n_components_-1):
                # find v_perp in span(v_remaining) such than 
                # span(v_) without v_perp has
                # the best (largest) worst-case explained variance

                # z is the linear combination of v_remaining that 
                # forms v_perp 
                z = torch.randn(self.n_components_-i, 1)
                z /= torch.norm(z)  # normalize z to have unit norm
                z.requires_grad_()

                optimizer = torch.optim.Adam([z], lr=lr)
                v_perp = None 
                v_keep = None # the remaining vectors after removing v_perp
                for _ in range(n_iters):
                    optimizer.zero_grad()

                    # create v_perp
                    z_norm = torch.norm(z)
                    v_perp = v_remaining @ z / z_norm

                    # find the orthogonal complement of v_perp within v_remaining
                    # to do so, we:
                    # a. QR decomposition of [z, I_{self.n_components_-i}] yields 
                    #    an orthonormal basis Q, where the first column is aligned
                    #    with z and the remaining columns are orthogonal to z
                    #    and span the orthogonal complement of z in R^{self.n_components_-i}
                    eye = torch.eye(self.n_components_-i)
                    Q, _ = torch.linalg.qr(torch.cat([z / z_norm, eye[:, 1:]], dim=1))
                    
                    # b. double check that the first column of Q is aligned with z
                    cos_sim = torch.nn.functional.cosine_similarity(
                        Q[:, 0].unsqueeze(1), z / torch.norm(z), dim=0)
                    assert torch.isclose(torch.abs(cos_sim), torch.tensor(1.0), atol=1e-5)

                    # c. the remaining columns of Q are orthogonal to z
                    #    thus v_remaining @ Q[:, 1:] is the projection of
                    #    v_remaining onto the orthogonal complement of z
                    z_perp = Q[:, 1:]  
                    v_keep = v_remaining @ z_perp  

                    # we can now compute the loss function and optimize
                    loss = f_minpca(v_keep, self.params_['covs'], self.params_['norm_csts'])
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    # save the results
                    assert torch.isclose(torch.norm(v_perp), torch.tensor(1.0), atol=1e-6)
                    v_ordered.append(v_perp.detach() / torch.norm(v_perp.detach()))
                    v_remaining = v_keep.detach()

                    if i == self.n_components_ - 2:
                        # last component, we need to add the last vector
                        v_ordered.append(v_keep.detach() / torch.norm(v_keep.detach()))

            v_ordered = torch.hstack(v_ordered)
            # flip to have the first component with the largest variance
            v_ordered = torch.flip(v_ordered, dims=[1]) 

            # check that the variance is the same as the original solution
            var = f_minpca(v_ordered, self.params_['covs'], 
                           self.params_['norm_csts']).item()
            old_var = f_minpca(self.v_, self.params_['covs'],
                               self.params_['norm_csts']).item()
            assert np.isclose(var, old_var), \
                f"Ordered PCA has different variance. New: {var:.4f} vs old: {old_var:.4f}"
            
            # update the attributes
            self.v_ = v_ordered
            self.cumsum_minvar_ = []
            self.cumsum_pooled_var_ = []
            for i in range(self.n_components_):
                var_i, pooled_var_i = get_vars_pca(self.v_[:, :i+1], self.params_, 
                                                self.params_pooled_)
                self.cumsum_minvar_.append(var_i)
                self.cumsum_pooled_var_.append(pooled_var_i)

        # return the components
        if isinstance(self.v_, torch.Tensor):
            return self.v_.detach().numpy()
        else:
            return self.v_


    def maxerr(self): 
        # TODO: also get index
        return self.maxerr_
    

    def pooled_err(self):
        return self.pooled_err_
    

    def minvar(self):
        return self.minvar_
    

    def pooled_var(self):
        return self.pooled_var_
