#!/usr/bin/env python3

import os
import sys
import warnings
from torch.jit import TracerWarning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XFORMERS_MORE_DETAILS"] = "0"
warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", message=".*.grad attribute of a Tensor that is not a leaf Tensor.*", category=UserWarning)

import argparse, math, asyncio
from io import BytesIO
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Body, Request, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_CH = 4
VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE).eval()

class _Enc(nn.Module):
    def __init__(self, vae): super().__init__(); self.vae=vae
    def forward(self,x): return self.vae.encode(x).latent_dist.mean
class _Dec(nn.Module):
    def __init__(self, vae): super().__init__(); self.vae=vae
    def forward(self,z): return self.vae.decode(z).sample
_ex = torch.randn(1,3,512,512,device=DEVICE)
scripted_encode = torch.jit.trace(_Enc(VAE).to(DEVICE),_ex,strict=False)
_lat_demo = scripted_encode(_ex)
scripted_decode = torch.jit.trace(_Dec(VAE).to(DEVICE),_lat_demo,strict=False)

def _pre(img: Image.Image):
    img=img.resize((512,512),Image.LANCZOS)
    arr=np.asarray(img,dtype=np.float32)/127.5-1.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)

def _encode(img_bytes: bytes, ds:int):
    img=Image.open(BytesIO(img_bytes)).convert("RGB")
    w,h=img.size
    with torch.no_grad():
        lat64=scripted_encode(_pre(img))
        lat_ds=F.interpolate(lat64,size=(ds,ds),mode="bilinear",align_corners=False)
    return [w,h]+lat_ds.cpu().view(-1).tolist()

def _decode(vec):
    if len(vec)>2 and isinstance(vec[0],(int,float)):
        w,h,*latent=vec
    else: w=h=None; latent=vec
    ds=int(math.sqrt(len(latent)/LATENT_CH))
    lat=torch.tensor(latent,dtype=torch.float32,device=DEVICE)\
        .view(1,LATENT_CH,ds,ds)
    with torch.no_grad():
        lat64=F.interpolate(lat,size=(64,64),mode="bilinear",align_corners=False)
        img_t=scripted_decode(lat64)
    arr=((img_t.cpu()[0].permute(1,2,0)+1)*127.5).clamp(0,255).byte().numpy()
    img=Image.fromarray(arr)
    if w and h: img=img.resize((w,h),Image.LANCZOS)
    buf=BytesIO(); img.save(buf,"JPEG",quality=85)
    return buf.getvalue()

# ──────────────────────────────────────────────────
#  FastFormer building block
# ──────────────────────────────────────────────────
class GlobalAdditiveAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d = d_model; self.h = n_heads; head_dim = d_model//n_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)
        self.scale = head_dim ** -0.5
    def forward(self,x):                # x:(B,N,d)
        B,N,_=x.shape
        q=self.q(x).view(B,N,self.h,-1)           # (B,N,h,dh)
        k=self.k(x).view(B,N,self.h,-1)
        v=self.v(x).view(B,N,self.h,-1)
        # global query / key
        q_g = (q.softmax(dim=1)*q).sum(dim=1,keepdim=True)   # (B,1,h,dh)
        k_g = (k.softmax(dim=1)*k).sum(dim=1,keepdim=True)
        # element-wise modulation
        y = v * (q_g * self.scale) + v * (k_g * self.scale)
        y = y.view(B,N,self.d)
        return self.fc(y)

class FastFormerBlock(nn.Module):
    def __init__(self,d_model=32,n_heads=4,ff=128):
        super().__init__()
        self.attn = GlobalAdditiveAttn(d_model,n_heads)
        self.norm1= nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(nn.Linear(d_model,ff),
                                  nn.GELU(),
                                  nn.Linear(ff,d_model))
        self.norm2= nn.LayerNorm(d_model)
    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        x=x+self.ff(self.norm2(x))
        return x

class LatentFastFormer(nn.Module):
    def __init__(self, ds, layers=2, d_model=32, n_heads=4):
        super().__init__()
        self.ds=ds; S=ds*ds
        self.inproj = nn.Linear(LATENT_CH,d_model)
        self.pos = nn.Parameter(torch.randn(1,S,d_model))
        self.blocks = nn.Sequential(
            *[FastFormerBlock(d_model,n_heads) for _ in range(layers)]
        )
        self.out = nn.Linear(d_model, LATENT_CH)
    def forward(self,x_flat):           # x_flat:(B,D)
        B,D = x_flat.shape
        tok = x_flat.view(B,LATENT_CH,self.ds,self.ds)\
                    .permute(0,2,3,1).reshape(B,-1,LATENT_CH) # (B,S,4)
        h = self.inproj(tok)+self.pos
        h = self.blocks(h)                                  # (B,S,d)
        out = self.out(h).reshape(B,self.ds,self.ds,LATENT_CH)\
                         .permute(0,3,1,2).reshape(B,D)
        return out

HISTORY=[]
MODEL=None; OPT=None; LOSS=nn.MSELoss()
LR=3e-4; EPOCHS=10; TRAIN_LOCK=asyncio.Lock()

def _split(vec):
    w,h,*body = vec if len(vec)>2 else (0,0,*vec)
    return (int(w),int(h)), torch.tensor(body,dtype=torch.float32,
                                         device=DEVICE)

async def _maybe_train():
    global MODEL, OPT
    if len(HISTORY)<2: return
    xs,ys = zip(*[(HISTORY[i][1],HISTORY[i+1][1]) for i in range(len(HISTORY)-1)])
    x=torch.stack(xs); y=torch.stack(ys)
    ds=int(math.sqrt(x.shape[1]/LATENT_CH))
    if MODEL is None or MODEL.ds!=ds:
        MODEL=LatentFastFormer(ds).to(DEVICE)
        OPT=torch.optim.AdamW(MODEL.parameters(),lr=LR)
    async with TRAIN_LOCK:
        MODEL.train()
        for _ in range(EPOCHS):
            OPT.zero_grad()
            pred=MODEL(x)
            loss=LOSS(pred,y)
            loss.backward(); OPT.step()
        MODEL.eval()

def _forecast(vec, want_jpeg=False):
    if MODEL is None or len(HISTORY)<2:
        raise HTTPException(400,"Need ≥2 decode calls first")
    (w,h),body=_split(vec)
    with torch.no_grad():
        nxt=MODEL(body.unsqueeze(0)).squeeze(0)
    next_vec=[w,h]+nxt.cpu().tolist()
    return (next_vec,_decode(next_vec)) if want_jpeg else (next_vec,None)


app=FastAPI(title="SD-VAE codec + FastFormer",docs_url=None,redoc_url=None)

@app.post("/encode")
async def encode_ep(request:Request,ds:int=64):
    ctype=request.headers.get("content-type","")
    if ctype.startswith("multipart/form-data"):
        form=await request.form(); up:UploadFile=form.get("file")
        if up is None: raise HTTPException(400,"No file")
        img_bytes=await up.read()
    else: img_bytes=await request.body()
    return JSONResponse(_encode(img_bytes,ds))

@app.post("/decode")
async def decode_ep(vec:list=Body(...)):
    try:
        shape,body=_split(vec); HISTORY.append((shape,body))
    except Exception as e:
        raise HTTPException(400,f"Invalid latent: {e}")
    jpg=_decode(vec); asyncio.create_task(_maybe_train())
    return Response(content=jpg,media_type="image/jpeg")

@app.post("/predict")
async def predict_ep(vec:list=Body(...,embed=True),jpeg:bool=False):
    next_vec,img_bytes=_forecast(vec,jpeg)
    if jpeg:
        return JSONResponse({"latent":next_vec,
                             "jpeg":"data:image/jpeg;base64,"+
                             BytesIO(img_bytes).getvalue().hex()})
    return JSONResponse(next_vec)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()
    print(f"Image Codec Ready on {DEVICE} – API: http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    uvicorn.run(app,host=args.host,port=args.port,log_level="warning")
