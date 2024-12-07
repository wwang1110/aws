from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import os
import subprocess

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism
from torch_xla.core.xla_model import is_master_ordinal


def training_function(script_args, training_args):
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle(seed=23)
    train_dataset = dataset.select(range(50000))
    eval_dataset = dataset.select(range(50000, 50500))

    def create_conversation(sample):
        system_message = (
            "You are a text to SQL query translator. Users will ask you questions in English and you will generate a "
            "SQL query based on the provided SCHEMA.\nSCHEMA:\n{schema}"
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_message.format(schema=sample["context"]),
                },
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"] + ";"},
            ]
        }

    train_dataset = train_dataset.map(
        create_conversation, remove_columns=train_dataset.features, batched=False
    )
    eval_dataset = eval_dataset.map(
        create_conversation, remove_columns=eval_dataset.features, batched=False
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.eos_token_id = 128001

    with lazy_load_for_parallelism(
        tensor_parallel_size=training_args.tensor_parallel_size
    ):
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id)

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(
        max_seq_length=1024,
        packing=True,
        **args,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    tokenizer_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "The tokenizer used to tokenize text for fine-tuning."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA r value to be used during fine-tuning."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha value to be used during fine-tuning."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value to be used during fine-tuning."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    training_function(script_args, training_args)

    # Consolidate LoRA adapter shards, merge LoRA adapters into base model, save merged model
    if is_master_ordinal():
        input_ckpt_dir = os.path.join(
            training_args.output_dir, f"checkpoint-{training_args.max_steps}"
        )
        output_ckpt_dir = os.path.join(training_args.output_dir, "merged_model")
        subprocess.run(
            [
                "python3",
                "consolidate_adapter_shards_and_merge_model.py",
                "-i",
                input_ckpt_dir,
                "-o",
                output_ckpt_dir,
            ]
        )
