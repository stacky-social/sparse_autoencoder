# Sparse autoencoders

This repository hosts:

- sparse autoencoders trained on the GPT2-small model's activations.
- a visualizer for the autoencoders' features

### Install

#### On MIT Engaging Computing Cluster

1. ssh into Engaging
2. Request a compute node for faster environment solving. NOTE: not everyone has access to this partition

```sh
# On a rocky 8 login node
srun --time 1:00:00 -n 1 -N 1 --partition mit_normal --pty /bin/bash
```

3. Set up a mamba environment with miniforge

```sh
module use /orcd/software/core/001/modulefiles/
module load miniforge/24.3.0-0 # to use mamba instead of conda!
mamba create --name stacky-sae python=3.10 pytorch triton blobfile
mamba activate stacky-sae
```

4. install the remaining pip libraries (transformer_lens, geom_median, dotenv, etc)

```sh
pip install -e . 
```

5. Create an environment file named `.env` and populate the following values:
   1. MILVUS_URI=<URI for Milvus db>
   2. MILVUS_TOKEN=<authentication token for Milvus db>
   3. LOG_FOLDER=<filepath to folder containing log files>
   4. EMBEDDINGS_FOLDER=<filepath folder containing embeddings>

6. Running the code: Request a compute node with a GPU. NOTE: not everyone has access to this partition

```sh
srun --time 1:00:00 --gres=gpu:1 -n 1 -N 1 --partition mit_normal_gpu --pty /bin/bash
module load miniforge
mamba activate stacky_sae
cd sparse_autoencoder
python testing.py #you should see that you have access to a CUDA GPU!
```

#### Failed attempts at environment setup:

1. ssh into Satori
2. set up your environment

```sh
# install custom miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh .
chmod +x ./Miniconda3-latest-Linux-ppc64le.sh
./Miniconda3-latest-Linux-ppc64le.sh -b -p ./mc3
source ./mc3/bin/activate

# add the relevant channels
conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/current

#rename the environment if you can't activate it 
conda rename -p path/to/mc3 mc3-sae
conda activate mc3-sae

#install packages!
conda install pytorch=2.1.2
conda install conda-forge::blobfile
#currently not able to get transformer_lens in here

#to run
srun --gres=gpu:1 -N 1 --mem=1T --time 1:00:00 --pty /bin/bash
conda activate mc3-sae
python testing.py
```

```sh
conda config --prepend channels \
https://opence.mit.edu
conda config --add channels conda-forge
export IBM_POWERAI_LICENSE_ACCEPT=yes
conda create --name stacky-sae python=3.7
conda install pytorch
conda install conda-forge::blobfile
conda install conda-forge::pyarrow
pip install transformer-lens

#to run
srun --gres=gpu:4 -N 1 --mem=1T --time 1:00:00 -I --pty /bin/bash
module load cuda/11.8.0
```

```sh
#trying a higher version of python
module load anaconda3/2020.02-2ks5tch
# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba
conda create --name stacky-sae python=3.10
conda activate stacky-sae
# module load cuda/11.4
conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/current
# conda install pytorch=1.12.1=cuda11.4_py310_1
# conda install pytorch=1.9.0=cuda10.2_py39_1
conda install pytorch=2.1.2=cuda12.2_py310_1 # will take forever
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

```sh
# experimenting
conda create --name stacky-sae2 python=3.12
conda activate stacky-sae2
pip install --upgrade pip
pip install --upgrade setuptools
pip install -e .
```

```sh
pip install -e .
```

### Code structure

See [sae-viewer](./sae-viewer/README.md) to see the visualizer code, hosted publicly [here](https://openaipublic.blob.core.windows.net/sparse-autoencoder/sae-viewer/index.html).

See [model.py](./sparse_autoencoder/model.py) for details on the autoencoder model architecture.
See [train.py](./sparse_autoencoder/train.py) for autoencoder training code.
See [paths.py](./sparse_autoencoder/paths.py) for more details on the available autoencoders.

### Example usage

```py
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(location, normalized_mse)
```
