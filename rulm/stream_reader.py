from typing import Dict, List, Iterable
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.language_modeling import LanguageModelingReader


@DatasetReader.register("lm_stream")
class LanguageModelingStreamReader(LanguageModelingReader):
    def __init__(self,
                 reverse: bool = False,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(tokens_per_instance, tokenizer, token_indexers, True)
        self.reverse = reverse

    @overrides
    def _read(self, file_path: str):
        for line in self._lines(file_path):
            if self._tokens_per_instance is None:
                yield self.text_to_instance(line)
                continue
            tokenized_text = self._tokenize(line)
            num_tokens = self._tokens_per_instance + 1
            for start in range(0, len(tokenized_text) - num_tokens, num_tokens - 1):
                end = start + num_tokens
                sample = tokenized_text[start:end]
                yield self._sample_to_instance(sample)

    def text_to_instance(self, text: str) -> Iterable[Instance]:
        return self._sample_to_instance(self._tokenize(text))

    def _tokenize(self, text: str) -> List[Token]:
        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text = tokenized_text[::-1] if self.reverse else tokenized_text
        tokenized_text.insert(0, Token(START_SYMBOL))
        tokenized_text.append(Token(END_SYMBOL))
        return tokenized_text

    def _sample_to_instance(self, sample: List[Token]) -> Instance:
        y = sample[1:]
        y.append(Token(DEFAULT_PADDING_TOKEN))
        input_field = TextField(sample, self._token_indexers)
        output_field = TextField(y, self._token_indexers)
        return Instance({
            'source_tokens': input_field,
            'target_tokens': output_field
        })

    @staticmethod
    def _lines(file_path: str) -> Iterable[str]:
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                line = line.strip()
                yield line