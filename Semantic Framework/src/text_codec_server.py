#!/usr/bin/env python3
"""
Contra-BT5 text codec
---------------------
* 512-dim latent  →  int8  (-4× size)
* Fully backward-compatible: reconstruction still uses float32.
"""
import argparse, logging, warnings, os, traceback
from typing import List

import torch
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# ── silence spam ────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "contra_bt5_small")       # local folder
EMB_SIZE   = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────
#  Bottleneck-T5 wrapper
# ────────────────────────────────────────────────────────────────
class BottleneckT5Autoencoder:
    def __init__(self, path: str, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, model_max_length=512, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.model.eval()

        # ---- PATCH: safe forward (handles cache_position, masks, etc.) -----
        cls              = self.model.__class__
        _orig_forward_fn = cls.forward
        self._orig_forward = _orig_forward_fn.__get__(self.model, cls)

        def _safe_forward(model_self, *args, **kw):
            kw.pop("cache_position", None)                         # 0️⃣ clean arg

            # 1️⃣ guarantee attention_mask
            if kw.get("attention_mask") is None:
                bs, seq_len = 1, 1
                enc_out = kw.get("encoder_outputs")
                if enc_out is not None and getattr(enc_out, "last_hidden_state", None) is not None:
                    seq_len = enc_out.last_hidden_state.shape[1]
                    bs      = enc_out.last_hidden_state.shape[0]
                else:
                    ids = kw.get("input_ids") or kw.get("decoder_input_ids")
                    if ids is not None:
                        seq_len = ids.shape[1]
                        bs      = ids.shape[0]
                kw["attention_mask"] = torch.ones(
                    bs, seq_len, dtype=torch.long, device=model_self.device
                )

            # 2️⃣ fabricate encoder_outputs if absent
            if kw.get("encoder_outputs") is None:
                vec     = getattr(model_self, "perturb_vector", None)
                d_model = model_self.config.d_model
                bs, seq_len = kw["attention_mask"].shape
                if vec is not None:
                    hid = vec.view(1, 1, -1).expand(bs, seq_len, -1)
                else:
                    hid = torch.zeros(bs, seq_len, d_model, device=model_self.device)
                from types import SimpleNamespace
                kw["encoder_outputs"] = SimpleNamespace(last_hidden_state=hid)

            return _orig_forward_fn(model_self, *args, **kw)

        cls.forward = _safe_forward
        # --------------------------------------------------------------------

    # ────────────────────────────────────────────────────────────────────
    #  Codec API
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        """Return 512-dim float32 latent for a **whole document**."""
        inp = self.tokenizer(text, return_tensors="pt").to(self.device)
        dec = self.tokenizer("",   return_tensors="pt").to(self.device)

        lat = self._orig_forward(
            **inp,
            decoder_input_ids=dec["input_ids"],
            encode_only=True
        )[0]                       # (1, 512)
        return lat.squeeze(0)      # (512,)

    @torch.no_grad()
    def generate(self, latent_f32: torch.Tensor, max_len=512, temp=1.0) -> str:
        """Reconstruct text from float32 latent."""
        dummy_txt  = "."
        base_lat   = self.embed(dummy_txt)
        self.model.perturb_vector = latent_f32 - base_lat
        ids = self.tokenizer(dummy_txt, return_tensors="pt").to(self.device).input_ids
        out = self.model.generate(
            input_ids=ids,
            max_length=max_len,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ────────────────────────────────────────────────────────────────
#  Helpers for int8 ⇄ float32
# ────────────────────────────────────────────────────────────────
def quantize_to_int8(lat_f32: torch.Tensor) -> List[int]:
    """Float32 (-1..1) → int8 list (-128..127)."""
    lat_clamped = lat_f32.clamp(-1, 1)          # safety
    return (lat_clamped * 127).round().to(torch.int8).tolist()

def dequantize_from_int8(lat_int8: List[int]) -> torch.FloatTensor:
    """int8 list → Float32 tensor on DEVICE."""
    arr = torch.tensor(lat_int8, dtype=torch.int8, device=DEVICE).to(torch.float32)
    return arr / 127.0


AE = BottleneckT5Autoencoder(MODEL_PATH, DEVICE)

# ────────────────────────────────────────────────────────────────
#  FastAPI service
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="Contra-BT5-small codec (int8 latent)",
              docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode(texts: List[str] = Body(..., embed=True)):
    """Return **int8** embeddings (arrays of 512 ints)."""
    try:
        out = [quantize_to_int8(AE.embed(t).cpu()) for t in texts]
        return JSONResponse(out)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Encode error: {e}")

@app.post("/decode")
async def decode(embeddings: List[List[int]] = Body(..., embed=True)):
    """Accept int8 embeddings, decode to text."""
    try:
        outs = []
        for vec in embeddings:
            if len(vec) != EMB_SIZE:
                raise HTTPException(400, f"Each embedding must be {EMB_SIZE} numbers")
            lat_f32 = dequantize_from_int8(vec)          # → float32
            outs.append(AE.generate(lat_f32))
        return JSONResponse(outs)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Decode error: {e}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"Codec ready → http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port,
                workers=1, log_level="warning")
