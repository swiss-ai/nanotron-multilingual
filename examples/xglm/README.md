# How to use XGLM?

1. First, make sure to convert the weights from huggingface, for instance:
   ```
   torchrun --nproc-per-node=1 examples/xglm/convert_hf2nt.py --checkpoint-path=facebook/xglm-564M --save-path=$SCRATCH/checkpoints/xglm-564M
   ```

1. Now you are ready to use XGLM.
   Make sure you use a .yaml configuration with proper GPT3 config and then run for instance:
   ```
   torchrun --nproc-per-node=4 run_train.py --config-file=examples/xglm/example_config.yaml
   ```
   If you use this configuration file make sure to modify at least the loading path in `model.init_method.path`.

1. If you want to convert your finetuned checkpoint back to huggingface use:
   ```
   torchrun --nproc-per-node=1 examples/xglm/convert_nt2hf.py --checkpoint-path=checpoints/xglm --save-path=$SCRATCH/checkpoints/huggingface/xglm-564M --tokenizer-name=facebook/xglm-564M
   ```
