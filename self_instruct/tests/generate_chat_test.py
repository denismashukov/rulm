from datasets import load_dataset
from src.data_processing.generate_chat import main
from pathlib import Path

self_instruct_dir = Path('../../self_instruct/external_prompts').resolve()
dataset_path = Path('../../.cache/dataset').resolve()
output_path = Path('../../.cache/output').resolve()


def test_fix_tokenizer():
    load_dataset_name = "IlyaGusev/ru_turbo_saiga"
    dataset = load_dataset(load_dataset_name)

    assert (dataset_path / "train_full.jsonl").exists()
    assert (dataset_path / "val_full.jsonl").exists()

    for split, split_dataset in dataset.items():
        split_dataset.to_json(dataset_path / f"squad-{split}.jsonl")

    main(dataset_path / 'squad-train.jsonl', output_path, self_instruct_dir / 'ru_chat.txt')

    assert (output_path).exists()

