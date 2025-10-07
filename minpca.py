import numpy as np
import torch
from minPCA.utils import get_solution, f_maxreconstruction, f_minpca, f_pca

def get_errs_pca(v, params=None, params_pooled=None):
    max_training = None 
    pooled = None
    if params is not None:
        max_training = f_maxreconstruction(v, params['Xs'], 
                                           params['norm_csts']).detach().item()
    if params_pooled is not None:
        pooled = f_maxreconstruction(v, params_pooled['Xs'], 
                                     params_pooled['norm_csts']).detach().item()
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


def generate_params(Xs, pooled=False, norm=True, from_cov=False):
    if isinstance(Xs[0], np.ndarray):
        Xs = [torch.from_numpy(X).float() for X in Xs]
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
        Xs_centred = [X - X.mean(dim=0) for X in Xs]
        covs = [X.t() @ X  for X in Xs_centred]

    # if norm is True, we normalize the covariance matrices by their trace
    if norm: 
        norm_csts = [torch.trace(cov) for cov in covs]
    else: 
        norm_csts = [1] * len(covs)

    params = {'Xs': Xs_centred, 'covs': covs, 'norm_csts': norm_csts}
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
        The function to optimize. One of ['minpca', 'minpca_pen', 
        'maxreconstruction', 'seq', 'fairpca', 'pooled', 'average']
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
        assert isinstance(n_components, int)
        assert function in ['minpca', 'minpca_pen', 'maxreconstruction',
                            'seq', 'fairpca', 'pooled', 'average', 'regret']
        assert isinstance(norm, bool)
        self.n_components_ = n_components
        self.function_ = function
        self.norm_ = norm

    def fit(self, Xs, n_iters=500, lr=0.1, betas=(0.9, 0.99), method='Adam',
            from_cov=False, n_restarts=5):
        assert isinstance(Xs, list)
        assert len(set(X.shape[1] for X in Xs)) == 1
        if (Xs[0].shape[0] == Xs[0].shape[1]) and not from_cov:
            print("Warning: the input data is square. This is not a problem, but ",
                  "it is not the usual case. \nIf you want to use covariance ",
                  "matrices, set `from_cov=True`.")
        if isinstance(Xs[0], np.ndarray):
            Xs = [torch.from_numpy(X).float() for X in Xs]

        self.K_ = len(Xs)
        self.p_ = Xs[0].shape[1]
        if from_cov: 
            self.Xs, self.means_, self.pooled_mean_ = None, None, None
        else: 
            self.Xs = Xs
            self.means_ = [X.mean(dim=0) for X in Xs]
            self.pooled_mean_ = torch.cat(Xs, dim=0).mean(dim=0)

        self.params_ = generate_params(Xs, norm=self.norm_, from_cov=from_cov)
        self.params_pooled_ = generate_params(Xs, pooled=True, norm=self.norm_,
                                              from_cov=from_cov)

        v0 = torch.randn(self.p_, self.n_components_)
        v0 = get_solution(v0, self.params_, function=self.function_, c=0.99, 
                               rank=self.n_components_, n_iters=n_iters,
                               lr=lr, betas=betas, method=method)
        best_v, best_var = v0, get_vars_pca(v0, self.params_)[0]
        if n_restarts > 1:
            for _ in range(n_restarts - 1):
                v_try = torch.randn(self.p_, self.n_components_)
                v_try = get_solution(v_try, self.params_, function=self.function_, c=0.99, 
                                  rank=self.n_components_, n_iters=n_iters,
                                  lr=lr, betas=betas, method=method)
                try_var = get_vars_pca(v_try, self.params_)[0]
                if best_var < try_var:
                    print(f"\timproving from {best_var:.3f} to {try_var:.3f}")
                    best_v, best_var = v_try, try_var
        self.v_ = best_v

        if from_cov:
            # TODO: can do error from the covariances!
            self.maxerr_, self.pooled_err_ = None, None
        else:
            self.maxerr_, self.pooled_err_ = get_errs_pca(
                self.v_, self.params_, params_pooled=self.params_pooled_)
        
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


    def reconstruct(self, X):
        if isinstance(X, torch.Tensor):
            X = [X]
        assert isinstance(X, list)
        X_means = [x.mean(dim=0) for x in X]
        X_centre = [x - mean for x, mean in zip(X, X_means)]    
        return [x @ self.v_ @ self.v_.t() + mean 
                for x, mean in zip(X_centre, X_means)]
    

    def maxerr(self): 
        # TODO: also get index
        return self.maxerr_
    
    def pooled_err(self):
        return self.pooled_err_
    
    def all_rcs_errs(self):
        # TODO
        return 
    
    def minvar(self):
        return self.minvar_
    
    def pooled_var(self):
        return self.pooled_var_
    
    def all_explained_vars(self, pooled=False):
        """
        Returns the explained variance for the rank i=1...n_components_ solutions.
        If `pooled` is True, returns the pooled explained variance.
        Otherwise, returns the minimum explained variance over the environments.
        """
        if pooled:
            return self.cumsum_pooled_var_
        else:
            return self.cumsum_minvar_

    # TODO: I DONT LIKE>>>FIX
    def get_rcs_errs(self, test_params=None):
        if test_params is None:
            return get_errs_pca(self.v_, self.params_, self.params_pooled_)
        else:
            return get_errs_pca(self.v_, **test_params)
    

    def get_explained_vars(self, test_params=None, all=True, rank=None):
        """
        If not `all`: Returns the minimum explained variance over the environments
            and the pooled explained variance.
        If `all`: Returns the explained variance for each environment
            for the rank n_components_ solution or given `rank`.
        """
        # get the minimum explained variance over the environments
        # and the pooled explained variance
        if not all:
            if test_params is None:
                return get_vars_pca(self.v_, self.params_, self.params_pooled_)
            else:
                return get_vars_pca(self.v_, **test_params)
            
        # if `all`: for each environment, get the explained variance
        else:
            if rank is None:
                rank = self.n_components_
            assert rank <= self.n_components_
            explained_vars = [
                f_pca(self.v_[:,:rank], cov, norm_cst).item()
                for cov, norm_cst in zip(self.params_['covs'], 
                                         self.params_['norm_csts'])
            ]
            return explained_vars

    # err1, _, err2 = get_errs_pca(v, params, params_pooled=params_pooled)
    # var1 = -f_minpca(v, params['covs'], params['norm_csts'])
    # var2 = -f_minpca(v, params_pooled['covs'], params_pooled['norm_csts'])