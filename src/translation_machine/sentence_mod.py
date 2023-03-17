
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from pathlib import Path

dir_file = Path(__file__).parent

def _create_sentence_type(vocabulary,tokenizer,is_destination_language):
    """each  """
    class Sentence:
        """
            class that represents a sentence in a language whose vacabulary is specified by 'vocab',
            in a tokenized form
            set as attribute of class
        """
        vocab = vocabulary
        _is_destination_language = is_destination_language
        _tokenizer = tokenizer
        @classmethod
        def to_int(cls,tokens):
            """convert tokens to int using the vocabulary"""
            res = [cls.vocab[token] for token in tokens]
            return res

        @classmethod
        def from_token_int(cls,tokens_int):
            sentence = " ".join(cls.vocab.itos_[idx] for idx in tokens_int if cls.vocab.itos_[idx] not in ['<unk>','<sos>','<eos>'])
            res = cls(sentence)
            return res

        def __init__(self,sentence) -> None:
            self._list_of_tokens = self._tokenizer(sentence)
            self._list_of_tokens_as_int = self.to_int(self._list_of_tokens)

        @property
        def list_of_tokens_as_int(self):
            return self._list_of_tokens_as_int
        @property
        def list_of_tokens(self):
            return self._list_of_tokens

        def __len__(self):
            return len(self._list_of_tokens)
        
        def pad(self,target):
            padded_list_of_tokens = self._list_of_tokens_as_int
            if self._is_destination_language:
                padded_list_of_tokens = [self.vocab['<sos>']] + padded_list_of_tokens + [self.vocab['<eos>']]
            padded_list_of_tokens = padded_list_of_tokens + [self.vocab['<unk>']]* (target - len(self._list_of_tokens))
            assert len(padded_list_of_tokens) == target + (2 if self._is_destination_language else 0)
            return padded_list_of_tokens
        
        def complete(self):
            assert self._is_destination_language,"valid only when the sentence is from the target language"
            padded_list_of_tokens = [self.vocab['<sos>']] + self._list_of_tokens_as_int + [self.vocab['<eos>']]
            return padded_list_of_tokens

        def is_complete(self):
            if self._is_destination_language:
                return self._list_of_tokens[-1] == self.vocab["<eos>"]
            else:
                raise ValueError(f"function is_complete should be called only if self._is_destination_language if True")

        def __str__(self) -> str:
            res = " ".join(self.as_words)
            return res

    
        @property
        def as_words(self):
            return self._list_of_tokens

        @property
        def as_int(self):
            tokens_list = self.to_int(self._list_of_tokens)
            return tokens_list

    return Sentence


    # def __gt__(self,other):
    #     return (self.proba > other.proba)
    # def __eq__(self,other):
    #     return self.proba == other.proba
    # def __iter__(self):
    #     return (el for el in (self._list_of_tokens,self.proba))


english_tokenizer = get_tokenizer('spacy', language='en')
french_tokenizer = get_tokenizer('spacy', language='fr')


path_language_info = str(dir_file.joinpath("../../models/language_info.pth"))
language_info = torch.load(path_language_info)
vocab_english = language_info["english"]["vocab"].vocab
EnglishSentence = _create_sentence_type(vocab_english,english_tokenizer,is_destination_language=False)

vocab_french = language_info["french"]["vocab"].vocab
FrenchSentence = _create_sentence_type(vocab_french,french_tokenizer,is_destination_language=True)