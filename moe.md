# MoE Env Setup

TL;DR: need to install megablocks for MoEs, which depends on triton; cannot install triton inside the docker image because it requires a CUDA-capable GPU, which is not available in the build environment. therefore install triton from source inside a venv in the container, then install megablocks


```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.10-venv && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update flash-attn.
RUN pip install --upgrade --no-build-isolation flash-attn==2.5.8

# Install the rest of dependencies.
RUN pip install \
    datasets \
    transformers \
    wandb \
    dacite \
    pyyaml \
    numpy \
    packaging \
    safetensors \
    tqdm 

```


after image is built, create env `~/.edf/nanotron-moe.toml` with content (adapt to wherever the image is stored)
```
image = "/capstor/scratch/cscs/$USER/container-images/nanotron-moe/nanotron-moe-v1.0.sqsh"

mounts = ["/capstor", "/users", "/store"]
workdir = "/users/$USER/"
writable = true
  
[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
  
[env]
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
NCCL_DEBUG = "INFO"
```

TODO: make image available on the cluster in /store



in a running container (`srun --reservation=todi --environment=nanotron-moe --container-workdir=$PWD --pty bash`)
```bash
cd $SCRATCH/$USER/nanotron-multilingual # or wherever you want the venv
mkdir multilingual-venv && cd multilingual-venv
python -m venv --system-site-packages ./moe-venv
source ./moe-venv/bin/activate
git clone https://github.com/triton-lang/triton.git; \
    cd triton; \
    pip install ninja cmake wheel; # build-time dependencies \
    pip install -e python; cd ..
pip install megablocks==0.5.1
```

