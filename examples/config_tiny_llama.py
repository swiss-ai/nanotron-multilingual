import os

from nanotron.config import (
    CheckpointsArgs,
    Config,
    DataArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

model_config = LlamaConfig(
    # Config for a tiny model model
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=16,
    initializer_range=0.02,
    intermediate_size=64,
    max_position_embeddings=256,
    num_attention_heads=4,
    num_hidden_layers=2,
    num_key_value_heads=4,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=2048,
)

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        3 * model_config.hidden_size * model_config.intermediate_size
        + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=10, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    adam_eps=1e-08,
    adam_beta1=0.9,
    adam_beta2=0.95,
    torch_adam_is_fused=True,
    learning_rate_scheduler=learning_rate,
)

dataset = PretrainDatasetsArgs(
    hf_dataset_or_datasets="HuggingFaceH4/testing_alpaca_small", text_column_name="completion"
)

checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="debug", run="tiny_llama", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
    parallelism=ParallelismArgs(1, 1, 1),
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("gpt2"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=TokensArgs(sequence_length=256, train_steps=100, micro_batch_size=4, batch_accumulation_per_replica=1),
    data=DataArgs(dataset=dataset, seed=seed),
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    config.save_as_yaml(f"{dir}/config_tiny_llama.yaml")
