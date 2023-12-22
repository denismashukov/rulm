from src.infer_saiga_llamacpp import infer
from pathlib import Path

content_dir = Path('../../.cache/content').resolve()


def infer_saiga_llamacpp_infer_test():
    max_new_tokens = 20  # @param {type:"integer"}

    model_name = "IlyaGusev/saiga_7b_ggml"

    infer(model_name, content_dir / 'val.jsonl', content_dir / 'test_result.jsonl', max_new_tokens)

    assert (content_dir / 'test_result.jsonl').exists()


infer_saiga_llamacpp_infer_test()
