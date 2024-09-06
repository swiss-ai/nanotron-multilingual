# How to use XGLM?

1. First, make sure to convert the weights from huggingface, for instance:
```bash
torchrun --nproc-per-node=1 examples/xglm/convert_hf2nt.py --checkpoint-path=facebook/xglm-564M --save-path=$SCRATCH/checkpoints/xglm-564M
```

2. Now you are ready to use XGLM.
   Make sure you use a .yaml configuration with proper GPT3 config and then run for instance:
```bash
torchrun --nproc-per-node=4 run_train.py --config-file=examples/xglm/example_config.yaml
```
   If you use this configuration file make sure to modify at least the loading path in `model.init_method.path`.

3. If you want to convert your finetuned checkpoint back to huggingface use:
```bash
torchrun --nproc-per-node=1 examples/xglm/convert_nt2hf.py --checkpoint-path=checkpoints/xglm --save-path=$SCRATCH/checkpoints/huggingface/xglm-564M --tokenizer-name=facebook/xglm-564M
```

## Sparse Upcycling

To create a sparse model from a dense model, you can use the `convert_dense2moe.py` script that goes from a GPT3 Nanotron model to a GPT3 MoE Nanotron model. For instance:
```bash
cd examples/xglm
torchrun --nproc-per-node=1 convert_dense2moe.py --checkpoint-path=checkpoints/xglm-564M --save-path=$SCRATCH/checkpoints/xglm-8x564M --num-experts=8
```
Note that this upcycling _drops_ the bias parameters of the MLP because the MegaBlocks implementation does not support bias parameters. While this is a limitation of the current implementation, the performance is quickly recovered after a few training steps.
