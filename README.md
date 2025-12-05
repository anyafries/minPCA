# minPCA

## Installation

First, create a Python environment:
```bash
python -m venv venv_minpca
source venv_minpca/bin/activate  # On Windows use `venv_minpca\Scripts\activate`
```

The code is organized as a Python package, and can be installed using `pip`:
```bash
git clone https://github.com/anyafries/minPCA.git
cd minPCA
pip install .
cd ..
```
To install it in editable mode (for modifying the code and seeing the changes immediately) and with developer dependencies (for testing and code formatting), use

```bash
pip install -e ".[dev]"
```

## Example use

Load libraries and functions
```python
import numpy as np
from minPCA.minpca import minPCA
from minPCA.utils import get_random_covs
```

Set-up 
```python
p = 20              # number of covariances
n_components = 5    # true rank of covariance
n_envs = 5          # number of environments

# Generate covariance matrices
covs = get_random_covs(p=p, rank=5, nenvs=n_envs)

# Sample from them
Xs = [np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=100000) 
      for cov in covs]
```

Solve minPCA
* `function` is one of 'minpca', 'maxreconstruction'
* argument `norm` can be set to True or False for the normalized versions (default)
```python
minpca = minPCA(n_components=n_components, function='maxreconstruction')
minpca = minpca.fit(covs, from_cov=True)

# errors/variance on the covariance matrices
print(f"Maximum (population) error: {minpca.maxerr():.3f}")
print(f"Minimum (population) variance: {minpca.minvar():.3f}")

# access the projection
components = minpca.v_.detach().numpy()
Xs_reconstructed = [Xs[i] @ components @ components.T for i in range(n_envs)]

# compute errors in finite sample
errs = [np.linalg.matrix_norm(Xs[i] - Xs_reconstructed[i], ord='fro')**2 / 100000 for i in range(n_envs)]
print(f"Maximum (sample) error: {max(errs):.3f}")
```