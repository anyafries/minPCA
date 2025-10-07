import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

# ----------------------------------------------------------------
# Helper functions (don't really belong here)
# ----------------------------------------------------------------

from minPCA.utils import f_maxreconstruction
from minPCA.autoencoder import rcs_errs

def get_errs_pca(v, params, params_test=None, params_pooled=None):
    max_training = f_maxreconstruction(v, params['Xs'], params['norm_csts']).item()
    max_test = None
    pooled = None
    if params_test is not None:
        max_test = f_maxreconstruction(v, params_test['Xs'], params_test['norm_csts']).item()
    if params_pooled is not None:
        pooled = f_maxreconstruction(v, params_pooled['Xs'], params_pooled['norm_csts']).item()
    return max_training, max_test, pooled


def print_errs(model_name, max_training, max_test=None, pooled=None):
    print(f"RCS error for {model_name} (max train):   {max_training:.4f}")
    if max_test is not None:
        print(f"RCS error for {model_name} (max test):    {max_test:.4f}")
    if pooled is not None:
        print(f"RCS error for {model_name} (pooled data): {pooled:.4f}")


def eval_model(model, model_name, params, params_test=None, params_pooled=None):
    pca_errs = rcs_errs(model, params)
    max_pca_err = max(pca_errs)
    out = (pca_errs,)
    log_args = (model_name, max_pca_err)

    if params_test is not None:
        pca_errs_test = rcs_errs(model, params_test)
        max_pca_err_test = max(pca_errs_test)
        out += (pca_errs_test,)
        log_args += (max_pca_err_test,)
    else:
        log_args += (None,)
    
    if params_pooled is not None:
        av_err = rcs_errs(model, params_pooled)[0]
        out += (av_err,)
        log_args += (av_err,)

    print_errs(*log_args)
    return out


# ----------------------------------------------------------------
# Load the MNIST dataset
# ----------------------------------------------------------------

def get_mnist_data(train_idx=range(5), test_idx=range(5, 10), log=False, return_means=False):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
    ])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    digit_dfs = {i: [] for i in range(10)}
    data_loader = DataLoader(mnist_data, batch_size=1000, shuffle=False)
    for images, labels in data_loader:
        for i in range(10):
            digit_images = images[labels == i]
            flattened_images = digit_images.view(digit_images.size(0), -1)
            digit_dfs[i].append(flattened_images) 

    if log:
        for i in range(10):
            digit_dfs[i] = torch.cat(digit_dfs[i], dim=0)
            print(f"DataFrame X{i} has shape: {digit_dfs[i].shape}")

    # get training, test, and pooled (training)
    Xs = [digit_dfs[i] for i in train_idx]
    Xs_test = [digit_dfs[i] for i in test_idx]
    X_pooled = torch.cat(Xs, dim=0)
    # center the data
    Xs_mean = [X.mean(dim=0) for X in Xs]
    Xs_test_mean = [X.mean(dim=0) for X in Xs_test]
    X_pooled_mean = X_pooled.mean(dim=0)
    Xs = [X - mean for X, mean in zip(Xs, Xs_mean)]
    Xs_test = [X - mean for X, mean in zip(Xs_test, Xs_test_mean)]
    X_pooled = X_pooled - X_pooled_mean

    covs = [X.t() @ X for X in Xs]
    norm_csts = [torch.trace(cov) for cov in covs]
    params = {'Xs': Xs, 'covs': covs, 'norm_csts': norm_csts}

    covs_test = [X.t() @ X for X in Xs_test]
    norm_csts_test = [torch.trace(cov) for cov in covs_test]
    params_test = {'Xs': Xs_test, 'covs': covs_test, 'norm_csts': norm_csts_test}

    cov_pooled = X_pooled.t() @ X_pooled
    norm_pooled = torch.trace(cov_pooled)
    params_pooled = {'Xs': [X_pooled], 'covs': [cov_pooled], 'norm_csts': [norm_pooled]}

    if return_means:
        params['Xs_mean'] = Xs_mean
        params_test['Xs_mean'] = Xs_test_mean
        params_pooled['Xs_mean'] = X_pooled_mean

    return params, params_test, params_pooled


# ----------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------

def plot_digits(Xs, Xs_reconstructed=None, title="", entry=0, figsize=(4, 1),
                means=None):
    num_digits = len(Xs)
    if Xs_reconstructed is not None:
        fig, axs = plt.subplots(2, num_digits, figsize=(figsize[0], 2*figsize[1]))
        for i, (X, Xr) in enumerate(zip(Xs, Xs_reconstructed)):
            Xplot = X[entry] + means[i] if means is not None else X[entry]
            Xplotr = Xr[entry] + means[i] if means is not None else Xr[entry]
            axs[0, i].imshow(Xplot.reshape(28, 28), cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(Xplotr.reshape(28, 28), cmap='gray')
            axs[1, i].axis('off')
    else:
        fig, axs = plt.subplots(1, num_digits, figsize=figsize)
        for i, X in enumerate(Xs):
            Xplot = X[entry] + means[i] if means is not None else X[entry]
            axs[i].imshow(Xplot.reshape(28, 28), cmap='gray')
            axs[i].axis('off')
        
    plt.suptitle(title)
    plt.show()