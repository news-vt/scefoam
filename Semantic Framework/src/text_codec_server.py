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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

# ── silence spam ────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

MODEL_PATH = "facebook/bart-large"
EMB_SIZE   = 1024    
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────
#  Bottleneck-T5 wrapper
# ────────────────────────────────────────────────────────────────
class T5Autoencoder:
    def __init__(self, path: str = "facebook/bart-large", device="cpu"):
        self.device = device

        # tokenizer stays the same
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, model_max_length=512
        )

        # ⚠️ Use the Seq2SeqLM class, not the CausalLM one
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path
        ).to(self.device)
        self.model.eval()

        # ── NEW ── set these two so generate() can reference them
        self.d_model = self.model.config.d_model                  # 512 for t5-small
        self.decoder_start_token_id = self.model.config.decoder_start_token_id


    # def _safe_forward(model_self, *args, **kw):
    #     kw.pop("cache_position", None)                         # 0️⃣ clean arg

    #     # 1️⃣ guarantee attention_mask
    #     if kw.get("attention_mask") is None:
    #         bs, seq_len = 1, 1
    #         enc_out = kw.get("encoder_outputs")
    #         if enc_out is not None and getattr(enc_out, "last_hidden_state", None) is not None:
    #             seq_len = enc_out.last_hidden_state.shape[1]
    #             bs      = enc_out.last_hidden_state.shape[0]
    #         else:
    #             ids = kw.get("input_ids") or kw.get("decoder_input_ids")
    #             if ids is not None:
    #                 seq_len = ids.shape[1]
    #                 bs      = ids.shape[0]
    #         kw["attention_mask"] = torch.ones(
    #             bs, seq_len, dtype=torch.long, device=model_self.device
    #         )

    #     # 2️⃣ fabricate encoder_outputs if absent
    #     if kw.get("encoder_outputs") is None:
    #         vec     = getattr(model_self, "perturb_vector", None)
    #         d_model = model_self.config.d_model
    #         bs, seq_len = kw["attention_mask"].shape
    #         if vec is not None:
    #             hid = vec.view(1, 1, -1).expand(bs, seq_len, -1)
    #         else:
    #             hid = torch.zeros(bs, seq_len, d_model, device=model_self.device)
    #         from types import SimpleNamespace
    #         kw["encoder_outputs"] = SimpleNamespace(last_hidden_state=hid)

    #     return _orig_forward_fn(model_self, *args, **kw)

    # cls.forward = _safe_forward
    # # --------------------------------------------------------------------

    # ────────────────────────────────────────────────────────────────────
    #  Codec API
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        """
        Return a 512-dim float32 latent for a whole document via mean-pool.
        (Same signature as before.)
        """
        # tokenize + encode
        inp = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        # get encoder hidden states: (1, seq_len, d_model)
        # enc = self.model.get_encoder()(**inp).last_hidden_state
        enc = self.model.model.encoder(**inp).last_hidden_state 
        # mean-pool over tokens → (1, d_model) → (d_model,)
        return enc.mean(dim=1).squeeze(0)
    
    @torch.no_grad()
    def generate(self,
                 latent_f32: torch.Tensor,
                 max_len: int = 512,
                 temp: float = 1.0
                ) -> str:
        """
        Reconstruct text from a 512-dim latent (float32).
        (Same signature as before.)
        """
        # expand latent to (1,1,d_model)
        expanded = latent_f32.view(1, 1, self.d_model)
        enc_out = BaseModelOutput(last_hidden_state=expanded)

        # start from the decoder_start token
        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]],
            device=self.device
        )

        out_ids = self.model.generate(
            encoder_outputs=enc_out,
            decoder_input_ids=decoder_input_ids,
            max_length=max_len,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

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


AE = T5Autoencoder(MODEL_PATH, DEVICE)

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
