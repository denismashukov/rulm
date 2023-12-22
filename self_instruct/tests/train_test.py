from src import train
from src.data_processing import create_short_chat_set
from pathlib import Path

import os

import json
from huggingface_hub import snapshot_download

self_instruct_dir = Path('../../self_instruct').resolve()
dataset_path = Path('../../.cache/dataset').resolve()
content_dir = Path('../../.cache/content').resolve()


def create_short_chat_set_download_test():
    create_short_chat_set.main(dataset_path / "train_full.jsonl", dataset_path / "val_full.jsonl")

    assert (dataset_path / "train_full.jsonl").exists()
    assert (dataset_path / "val_full.jsonl").exists()


def train_test():
    model_dir = content_dir / "llama2-7b-chat-ht"
    base_model = "meta-llama/Llama-2-7b-chat-hf"  # meta-llama/Llama-2-7b-chat-hf
    snapshot_download(repo_id=base_model, local_dir=model_dir, token=os.getenv('HF_TOKEN'),
                      ignore_patterns=["LICENSE", "README.md", ".gitattributes"])

    patch_model_config = True  # @param {type:"boolean"}

    if patch_model_config:
        replacements = {
            "tokenizer_config.json": {
                "tokenizer_class": "LlamaTokenizer",
                "model_max_length": 4096,
                "padding_side": "left",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "clean_up_tokenization_spaces": False,
                "special_tokens_map_file": "special_tokens_map.json",
            },
            "special_tokens_map.json": {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
                "sep_token": "<s>",
                "unk_token": "<unk>",
            }
        }

        print('Patching model config...')
        for filename, new_content in replacements.items():
            print(f'{filename}:')
            with (model_dir / filename).open(encoding="utf8") as fp:
                old_content = json.load(fp)
                print(f'    Original content: {old_content}')
                if old_content == new_content:
                    print('    Already patched, skipping')
            print(f'    Updated content:  {new_content}')
            with (model_dir / filename).open('w', encoding="utf8") as fp:
                json.dump(new_content, fp, indent=4)

    original_config_path = self_instruct_dir / 'configs/saiga2_7b.json'

    with original_config_path.open('r') as fp:
        config = json.load(fp)

        # Colab adjustments
    config['trainer']['per_device_train_batch_size'] = 2  # @param {type:"integer"}
    config['trainer']['gradient_accumulation_steps'] = 64  # @param {type:"integer"}
    config['max_tokens_count'] = 1024  # @param {type:"integer"}
    config['model_name'] = str(model_dir)

    # Demo adjustments
    config['trainer']['eval_steps'] = 2  # @param {type:"integer"}
    config['trainer']['logging_steps'] = 1  # @param {type:"integer"}
    config['trainer']['num_train_epochs'] = 1  # @param {type:"integer"}

    config_path = self_instruct_dir / 'configs/saiga2_7b_colab.json'

    with config_path.open('w', encoding="utf8") as fp:
        json.dump(config, fp, indent=4)

    output_dir = content_dir / 'output'

    config_file = config_path.relative_to(self_instruct_dir)

    train.train(config_file, content_dir / 'train.jsonl', content_dir / 'val.jsonl', output_dir)

    assert (output_dir / 'adapter_config.json').exists()


train_test()
