"""
SAE architecture modelled after 
https://github.com/jsalt2024-evaluating-llms-for-astronomy/retrieval/blob/main/saerch/topk_sae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FastAutoencoder(nn.Module):
    def __init__(self, n_dirs: int, d_model: int, k: int, auxk: int, multik: int, dead_steps_threshold: int = 266):
        super().__init__()
        self.n_dirs = n_dirs # num dimensions in latent space
        self.d_model = d_model # num dimentions in embeddings
        self.k = k # num of top-k latents to keep
        self.auxk = auxk # num of top-auxk dead latents to keep, for auxiliary loss
        self.multik = multik # num of top-4k latents to keep, for calculating multik loss
        self.dead_steps_threshold = dead_steps_threshold # num of steps before a latent is considered dead

        self.encoder = nn.Linear(d_model, n_dirs, bias=False)
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs)) # (n_dirs,)

        self.stats_last_nonzero = torch.zeros(n_dirs, dtype=torch.long, device=device) # (n_dirs,)

    def forward(self, x, theta = 0.1):
        x = x - self.pre_bias # (batch_size, d_model)
        latents_pre_act = self.encoder(x) + self.latent_bias # (batch_size, n_dirs)

        # Main top-k selection
        topk_values, topk_indices = torch.topk(latents_pre_act, k=self.k, dim=-1) # each one is (batch_size, k)
        topk_values = F.relu(topk_values) # (batch_size, k)

        multik_values, multik_indices = torch.topk(latents_pre_act, k=4*self.k, dim=-1)
        multik_values = F.relu(multik_values)

        # only create reconstructions using the top k latent dimensions
        latents = torch.zeros_like(latents_pre_act) # (batch_size, n_dirs)
        latents.scatter_(-1, topk_indices, topk_values)
        multik_latents = torch.zeros_like(latents_pre_act) # (batch_size, n_dirs)
        multik_latents.scatter_(-1, multik_indices, multik_values)

        # Update stats_last_nonzero
        self.stats_last_nonzero += 1
        # Reset stats_last_nonzero for topk_indices because they just activated
        self.stats_last_nonzero.scatter_(0, topk_indices.unique(), 0)
 
        recons = self.decoder(latents) + self.pre_bias # (batch_size, d_model)
        multik_recons = self.decoder(multik_latents) + self.pre_bias # (batch_size, d_model)

        latents_jump = latents_pre_act.detach().clone()
        latents_jump[latents_jump < theta] = 0

        # AuxK
        if self.auxk is not None:
            # Create dead latents mask
            dead_mask = (self.stats_last_nonzero > self.dead_steps_threshold).float()
            
            # Apply mask to latents_pre_act
            dead_latents_pre_act = latents_pre_act * dead_mask
            
            # Select top-k_aux from dead latents
            auxk_values, auxk_indices = torch.topk(dead_latents_pre_act, k=self.auxk, dim=-1)
            auxk_values = F.relu(auxk_values)

        else:
            auxk_values, auxk_indices = None, None

        return recons, {
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_indices": multik_indices,
            "multik_values": multik_values,
            "multik_recons": multik_recons,
            "auxk_indices": auxk_indices,
            "auxk_values": auxk_values,
            "latents_pre_act": latents_pre_act,
            "latents_jump": latents_jump,
        }

    def decode_sparse(self, indices, values):
        latents = torch.zeros(self.n_dirs, device=indices.device)
        latents.scatter_(-1, indices, values)
        return self.decoder(latents) + self.pre_bias

    # def decode_clamp(self, latents, clamp):
    #     topk_values, topk_indices = torch.topk(latents, k = 64, dim=-1)
    #     topk_values = F.relu(topk_values)
    #     latents = torch.zeros_like(latents)
    #     latents.scatter_(-1, topk_indices, topk_values)
    #     # multiply latents by clamp, which is 1D but has has the same size as each latent vector
    #     latents = latents * clamp
        
    #     return self.decoder(latents) + self.pre_bias
    
    # def decode_at_k(self, latents, k):
    #     topk_values, topk_indices = torch.topk(latents, k=k, dim=-1)
    #     topk_values = F.relu(topk_values)
    #     latents = torch.zeros_like(latents)
    #     latents.scatter_(-1, topk_indices, topk_values)
        
    #     return self.decoder(latents) + self.pre_bias


############## GLOBAL FUNCTIONS ###############
def unit_norm_decoder_(autoencoder: FastAutoencoder) -> None:
    with torch.no_grad():
        autoencoder.decoder.weight.div_(autoencoder.decoder.weight.norm(dim=0, keepdim=True))

def unit_norm_decoder_grad_adjustment_(autoencoder: FastAutoencoder) -> None:
    if autoencoder.decoder.weight.grad is not None:
        with torch.no_grad():
            proj = torch.sum(autoencoder.decoder.weight * autoencoder.decoder.weight.grad, dim=0, keepdim=True)
            autoencoder.decoder.weight.grad.sub_(proj * autoencoder.decoder.weight)

def mse(output, target):
    return F.mse_loss(output, target)

def normalized_mse(recon, xs):
    return mse(recon, xs) / mse(xs.mean(dim=0, keepdim=True).expand_as(xs), xs)

def loss_fn(ae, x, recons, info, auxk_coef, multik_coef):
    """
    Compute a more detailed loss function for the autoencoder model
    total loss = normalized mse loss + normalized mse loss with 4*k + mornalized mse between actual error and dead latents error
    Args:
        ae: the autoencoder model
        x: input embeddings
        recons: reconstructed embeddings
        info: dictionary returned by the forward pass on the autoencoder model
        auxk_coef: coefficient for the auxiliary loss
        multik_coef: coefficient for the multik loss

    """
    recons_loss = normalized_mse(recons, x)
    recons_loss += multik_coef * normalized_mse(info["multik_recons"], x)
    
    if ae.auxk is not None:
        e = x - recons.detach()  # reconstruction error
        auxk_latents = torch.zeros_like(info["latents_pre_act"])
        auxk_latents.scatter_(-1, info["auxk_indices"], info["auxk_values"])
        e_hat = ae.decoder(auxk_latents)  # reconstruction of error using dead latents
        auxk_loss = normalized_mse(e_hat, e)
        total_loss = recons_loss + auxk_coef * auxk_loss
    else:
        auxk_loss = torch.tensor(0.0, device=device)
        total_loss = recons_loss
    
    return total_loss, recons_loss, auxk_loss

def init_from_data_(ae, data_sample):
    # set pre_bias to median of data
    ae.pre_bias.data = torch.median(data_sample, dim=0).values
    nn.init.xavier_uniform_(ae.decoder.weight)

    # decoder is unit norm
    unit_norm_decoder_(ae)

    # encoder as transpose of decoder
    ae.encoder.weight.data = ae.decoder.weight.t().clone()

    nn.init.zeros_(ae.latent_bias)

def train(ae, train_loader, optimizer, epochs, k, auxk_coef, multik_coef, clip_grad=None, save_dir="checkpoints", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    num_batches = len(train_loader)
    for epoch in range(epochs):
        ae.train() # sets the model to train mode
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            x = batch['embedding'].to(device)
            recons, info = ae(x)
            loss, recons_loss, auxk_loss = loss_fn(ae, x, recons, info, auxk_coef, multik_coef)
            loss.backward()
            step += 1

            # calculate proportion of dead latents (not fired in last num_batches = 1 epoch)
            dead_latents_prop = (ae.stats_last_nonzero > num_batches).float().mean().item()

            wandb.log({
                "total_loss": loss.item(),
                "reconstruction_loss": recons_loss.item(),
                "auxiliary_loss": auxk_loss.item(),
                "dead_latents_proportion": dead_latents_prop,
                "l0_norm": k,
                "step": step
            })
            
            unit_norm_decoder_grad_adjustment_(ae)
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), clip_grad)
            
            optimizer.step()
            unit_norm_decoder_(ae)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Delete previous model saves for this configuration
        for old_model in glob.glob(os.path.join(save_dir, f"{model_name}_epoch_*.pth")):
            os.remove(old_model)

        # Save new model
        save_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(ae.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def get_topk_activations(ae, embeddings, model_name, sae_data_path):
    """
        Args: 
            ae: the autoencoder model
            embeddings: all dataset embeddings in order
            model_name: the name of the ae as described upon initialization
            sae_data_path: the filepath to save the topk_indices and topk_values matrices
        Returns:
            None, saves two numpy files named {model_name}_topk_indices.npy and {model_name}_topk_values.npy
    """
    topk_indices = []
    topk_values = []
    for embed in embeddings:
        recons, info = ae(embed.to(device))
        # print(type(info["topk_indices"].cpu().size))
        topk_indices.append(info["topk_indices"].cpu())
        topk_values.append(info["topk_values"].cpu())
    
    # convert the list of tensors into a tensor
    topk_indices = torch.vstack(topk_indices)
    topk_values = torch.vstack(topk_values)

    # convert the tensor into a numpy array
    topk_indices = topk_indices.cpu().detach().numpy()
    topk_values = topk_values.cpu().detach().numpy()

    print(topk_indices.size)
    print(topk_values.size)

    if not os.path.exists(sae_data_path):
        os.makedirs(sae_data_path)

    indices_path = os.path.join(sae_data_path, f"{model_name}_topk_indices.npy")
    values_path = os.path.join(sae_data_path, f"{model_name}_topk_values.npy")
    np.save(indices_path, topk_indices)
    np.save(values_path, topk_values)

def main():
    d_model = 1536
    n_dirs = 9216 # 1536 * 6
    k = 64
    auxk = k*2 #256
    multik = 128
    batch_size = 1024
    lr = 1e-4
    epochs = 50
    auxk_coef = 1/32
    clip_grad = 1.0
    multik_coef = 0 # turn it off
    data_size = "full" #full, test
    init_type = "initsample10k" #initfull, initsample10k, initfirst10k

    load_dotenv()

    dataset = EmbeddingsDataset()
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), data_size)
    dataset.load_all_embeddings(dir=path, from_dir=True)
    embeddings = dataset.embeddings
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # the data loader

    ksae = FastAutoencoder(n_dirs, d_model, k, auxk, multik).to(device)
    
     # Initialize pre_bias to median of data
    if init_type == "initfull":
        data_sample = embeddings
    elif init_type == "initfirst10k":
        data_sample = embeddings[:10000]
    elif init_type == "initsample10k":
        #randomly sample from the embeddings matrix
        idx = torch.randperm(embeddings.shape[0]) 
        print(idx)
        shuffled_embeddings = embeddings[idx].view(embeddings.size())
        data_sample = shuffled_embeddings[:10000]

    init_from_data_(ksae, data_sample.to(device))

    optimizer = optim.Adam(ksae.parameters(), lr=lr)

    # Create model name
    model_name = f"k{k}_ndirs{n_dirs}_auxk{auxk}_{data_size}_{init_type}"

    wandb.init(project="topk_sae", name=model_name, config={
        "n_dirs": n_dirs,
        "d_model": d_model,
        "k": k,
        "auxk": auxk,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "auxk_coef": auxk_coef,
        "multik_coef": multik_coef,
        "clip_grad": clip_grad,
        "device": device.type,
        "data_size": data_size,
        "init_type": init_type
    })

    train(ksae, dataloader, optimizer, epochs, k, auxk_coef, multik_coef, clip_grad=clip_grad, model_name=model_name)

    wandb.finish()

if __name__ == "__main__":
    # main()
    d_model = 1536
    n_dirs = 9216 # 1536 * 6
    k = 64
    auxk = k*2 #256
    multik = 128
    batch_size = 1024
    lr = 1e-4
    epochs = 50
    auxk_coef = 1/32
    clip_grad = 1.0
    multik_coef = 0 # turn it off
    data_size = "full" #full, test
    init_type = "initsample10k" #initfull, initsample10k, initfirst10k

    load_dotenv()

    dataset = EmbeddingsDataset()
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), data_size)
    dataset.load_all_embeddings(dir=path, from_dir=True)
    embeddings = dataset.embeddings

    model_name = f"k{k}_ndirs{n_dirs}_auxk{auxk}_{data_size}_{init_type}"
    
    ksae = FastAutoencoder(n_dirs, d_model, k, auxk, multik).to(device)
    sae_data_path = os.path.join(os.environ.get("SAE_DATA_FOLDER"))
    get_topk_activations(ksae, embeddings, model_name, sae_data_path)