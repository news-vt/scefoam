#!/usr/bin/env python3
import torch
from diffusers import AutoencoderKL

# 1. Load VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

# 2. Build wrappers
class EncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, x):
        # returns the mean of the latent distribution
        return self.vae.encode(x).latent_dist.mean

class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, lat):
        # returns the decoded image tensor
        return self.vae.decode(lat).sample

enc_model = EncoderWrapper(vae).to(device).eval()
dec_model = DecoderWrapper(vae).to(device).eval()

# 3. Dummy inputs for tracing
x = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)
z = torch.randn(1, 4, 64, 64,    device=device, dtype=torch.float32)

# 4. Export encoder
torch.onnx.export(
    enc_model, x, "vae_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["latent_mean"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "latent_mean":   {0: "batch"}
    },
    opset_version=17,
    do_constant_folding=True,
)

# 5. Export decoder
torch.onnx.export(
    dec_model, z, "vae_decoder.onnx",
    input_names=["latent"],
    output_names=["reconstructions"],
    dynamic_axes={"latent": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
)

print("✅ Wrote vae_encoder.onnx and vae_decoder.onnx")
