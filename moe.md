# MoE Env Setup

TL;DR: need to install megablocks for MoEs. just use the environment `/store/swissai/a06/containers/nanotron_moe/nanotron_moe.toml` :)

The setup is documented in that folder on the cluster. The setup is:

```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3

# setup
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git tmux htop nvtop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools==69.5.1

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
    sentencepiece \
    tqdm 

WORKDIR /workspace
RUN git clone https://github.com/swiss-ai/nanotron.git
WORKDIR /workspace/nanotron
RUN pip install -e .[nanosets]

RUN pip install megablocks==0.5.1 stanford-stk==0.7.1 --no-deps
```

The env `nanotron-moe.toml` with content:
```
image = "/store/swissai/a06/containers/nanotron_moe/nanotron_moe.sqsh"

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
