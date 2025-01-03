"""
SAE architecture modelled after 
https://github.com/jsalt2024-evaluating-llms-for-astronomy/retrieval/blob/main/saerch/vanilla_sae.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sparse_autoencoder.embeddings import EmbeddingsDataset
from dotenv import load_dotenv
from tqdm import tqdm
import wandb
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaSparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, n_dirs: int):
        super().__init__()
        self.d_model = d_model # num dimensions in embeddings
        self.n_dirs = n_dirs # num dimensions in the latent spaceS
        
        self.encoder = nn.Linear(d_model, n_dirs) 
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model)) # (d_model,)
        
        self.stats_last_nonzero = torch.zeros(n_dirs, dtype=torch.long, device=device) # (n_dirs,)
        
    def forward(self, x):
        x = x - self.pre_bias # (batch_size, d_model)

        # a feature only "fires" if it is greater than 0
        z = torch.relu(self.encoder(x)) # (batch_size, n_dirs)
        x_hat = self.decoder(z) + self.pre_bias # (batch_size, d_model)
        
        # Update stats_last_nonzero
        self.stats_last_nonzero += 1
        # Reset stats_last_nonzero for features that fired (z[i] > 0)
        self.stats_last_nonzero[z.sum(dim=0) > 0] = 0
        
        return x_hat, z

def mse(output, target):
    return torch.mean((output - target) ** 2)

def normalized_mse(recon, xs):
    """
    Normalized MSE: MSE divided by MSE of mean
    Used as reconstruction loss
    """
    return mse(recon, xs) / mse(xs.mean(dim=0, keepdim=True).expand_as(xs), xs)

def loss_fn(x, x_hat, z, l1_lambda):
    """
    The loss function used to train the model
    total_loss = reconstruction_loss + l1_loss
    """
    reconstruction_loss = normalized_mse(x_hat, x)
    l1_loss = l1_lambda * z.norm(p=1, dim=-1).mean() #torch.mean(torch.abs(z))
    return reconstruction_loss + l1_loss, reconstruction_loss, l1_loss

def train(sae, train_loader, optimizer, epochs, l1_lambda, clip_grad=None, save_dir="checkpoints", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    num_batches = len(train_loader)
    for epoch in range(epochs):
        sae.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            x = batch['embedding'].to(device)
            x_hat, z = sae(x)
            loss, reconstruction_loss, l1_loss = loss_fn(x, x_hat, z, l1_lambda)
            loss.backward()
            step += 1
            
            # L0 norm (proportion of non-zero elements)
            l0_norm = torch.mean((z != 0).sum(dim=1).float()).item()
            
            # proportion of dead latents (not fired in last num_batches = 1 epoch)
            dead_latents_prop = (sae.stats_last_nonzero > num_batches).float().mean().item()
            
            wandb.log({
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "l1_loss": l1_loss.item(),
                "l0_norm": l0_norm,
                "dead_latents_proportion": dead_latents_prop,
                "step": step
            })
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Delete previous model saves for this configuration
        for old_model in glob.glob(os.path.join(save_dir, f"{model_name}_epoch_*.pth")):
            os.remove(old_model)

        # Save new model
        save_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(sae.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def main():
    d_model = 1536
    n_dirs = d_model * 8
    batch_size = 1024
    lr = 1e-4
    epochs = 20
    l1_lambda = 0.05
    clip_grad = 1.0
    data_size = "full" #full, test
    init_type = "initsample10k" #initfull, initsample10k, initfirst10k

    load_dotenv()


    dataset = EmbeddingsDataset()
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), data_size)
    dataset.load_all_embeddings(dir=path, from_dir=True)
    embeddings = dataset.embeddings
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # the data loader

    sae = VanillaSparseAutoencoder(d_model, n_dirs).to(device)
    
    # Initialize pre_bias to median of data
    with torch.no_grad():
        if init_type == "initfull":
            sae.pre_bias.data = torch.median(embeddings, dim=0)[0].to(device)
        elif init_type == "initfirst10k":
            sae.pre_bias.data = torch.median(embeddings[:10000], dim=0)[0].to(device)
        elif init_type == "initsample10k":
            #randomly sample from the embeddings matrix
            idx = torch.randperm(embeddings.shape[0]) 
            print(idx)
            shuffled_embeddings = embeddings[idx].view(embeddings.size())
            sae.pre_bias.data = torch.median(shuffled_embeddings[:10000], dim=0)[0].to(device)

    optimizer = optim.Adam(sae.parameters(), lr=lr)
            
    # Create model name
    model_name = f"vanilla_{n_dirs}_{data_size}_{init_type}"
    
    wandb.init(project="sae", name=model_name, config={
        "n_dirs": n_dirs,
        "d_model": d_model,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "l1_lambda": l1_lambda,
        "clip_grad": clip_grad,
        "device": device.type,
        "data_size": data_size,
        "init_type": init_type
    })

    train(sae, dataloader, optimizer, epochs, l1_lambda, clip_grad, model_name=model_name)

    wandb.finish()

if __name__ == "__main__":
    main()