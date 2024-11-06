import torch
import sparse_autoencoder
import time
import blobfile as bf
import transformer_lens

import torch
device_id = torch.cuda.current_device()
gpu_properties = torch.cuda.get_device_properties(device_id)

print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
      "%.1fGb total memory.\n" %
      (torch.cuda.device_count(),
       device_id,
       gpu_properties.name,
       gpu_properties.major,
       gpu_properties.minor,
       gpu_properties.total_memory / 1e9))

print("Extracting neuron activations with transformer_lens...")
tic = time.perf_counter()
# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device
print("device:", device)
toc = time.perf_counter()
print(f"Extracted neuron activations in {toc - tic:0.4f} seconds")

print("Cuda available: ", torch.cuda.is_available())
print("getting activation cache...")
tic = time.perf_counter()
prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
toc = time.perf_counter()
print(f"Got activation cache in {toc - tic:0.4f} seconds")

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

print("loading autoencoder...")
tic = time.perf_counter()
with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)
toc = time.perf_counter()
print(f"Loaded autoencoder in {toc - tic:0.4f} seconds")

input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

print("encoding and decoding...")
tic = time.perf_counter()
with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
toc = time.perf_counter()
print(f"Encoded and decoded in {toc - tic:0.4f} seconds")
print(location, normalized_mse)