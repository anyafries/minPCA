import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------
# Autoencoder model
# ----------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder=None, decoder=None, 
                 nlayers=1, skips=False):
        super(Autoencoder, self).__init__()
        self.skips = skips
        self.relu = nn.ReLU()

        if nlayers > 1:
            decrease = (input_dim - hidden_dim) // nlayers
            layer_dims = [input_dim - i * decrease for i in range(nlayers + 1)]
            hidden_dim = layer_dims[-1]

        if encoder is None:
            if nlayers > 1:
                enc_layers = []
                for i in range(nlayers):
                    enc_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                    if i < nlayers - 1:
                        enc_layers.append(self.relu)
            else: 
                enc_layers = [nn.Linear(input_dim, hidden_dim)]
            encoder = nn.Sequential(*enc_layers)
        if decoder is None:
            if nlayers > 1:
                dec_layers = []
                for i in range(nlayers-1, -1, -1):
                    dec_layers.append(nn.Linear(layer_dims[i + 1], layer_dims[i]))
                    if i > 0:
                        dec_layers.append(self.relu)
            else:
                dec_layers = [nn.Linear(hidden_dim, input_dim)]
            decoder = nn.Sequential(*dec_layers)

        self.encoder = encoder
        self.decoder = decoder
        
        if self.skips:
            self.skip_encode = nn.Linear(input_dim, hidden_dim)
            self.skip_decode = nn.Linear(hidden_dim, input_dim)
        

    def forward(self, x):
        if self.skips:
            encoded = self.encoder(x) + self.skip_encode(x)
            encoded = self.relu(encoded)
            # decoded = self.relu(self.decoder(encoded) + self.skip_decode(encoded))
            decoded = self.decoder(encoded) + self.skip_decode(encoded)
        else: 
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        return decoded


# ----------------------------------------------------------------
# Training function
# ----------------------------------------------------------------

def train_autoencoder(autoencoder, datasets, num_epochs=1000, lr=0.1, log=True, log_mod=100, 
                      verbose=False, get_losses=False):
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    dataset_norms = [frobenius_norm_squared(data) for data in datasets]

    all_dataset_losses = []
    all_max_dataset_losses = []

    for epoch in range(num_epochs):
        dataset_losses = []
        
        for i, data in enumerate(datasets):
            inputs = data 
            outputs = autoencoder(inputs)
            dataset_loss = frobenius_norm_squared(inputs - outputs) / dataset_norms[i]
            if verbose: 
                print(f'Dataset {i + 1} Loss: {dataset_loss.item():.4f}')
            dataset_losses.append(dataset_loss.unsqueeze(0).unsqueeze(1))

        max_dataset_loss = torch.max_pool1d(torch.cat(dataset_losses, dim=1), 
                                            kernel_size=len(datasets))[0,0] 

        # Print the gradients for each parameter
        if verbose:
            print("Gradients of each parameter:")
            for name, param in autoencoder.named_parameters():
                if param.grad is not None:
                    print(f"{name}.grad:\n{param.grad}")
                else:
                    print(f"{name} has no gradient.")

        # Backward pass and optimization
        optimizer.zero_grad()
        max_dataset_loss.backward()
        optimizer.step()

        # Logging for the current epoch
        if log & ((epoch+1) % log_mod == 0): 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {max_dataset_loss.item():.4f}')

        all_dataset_losses.append(dataset_losses)
        all_max_dataset_losses.append(max_dataset_loss.item())

    if get_losses: return all_dataset_losses, all_max_dataset_losses
    

# ----------------------------------------------------------------
# Evaluation functions
# ----------------------------------------------------------------

# Function to calculate Frobenius norm
def frobenius_norm_squared(tensor):
    return torch.norm(tensor, p='fro')**2


# Function to calculate the relative reconstruction error
def rcs_err(autoencoder, X, norm_cst):
    err =  frobenius_norm_squared(X - autoencoder(X)) 
    return err / norm_cst


# Function to calculate the relative reconstruction errors for all datasets
def rcs_errs(autoencoder, params):
    num_datasets = len(params['Xs'])
    errs = [rcs_err(autoencoder, params['Xs'][i], params['norm_csts'][i]).detach().item() 
            for i in range(num_datasets)]
    return errs