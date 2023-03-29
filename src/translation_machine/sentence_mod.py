
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from pathlib import Path

dir_file = Path(__file__).parent

def _create_sentence_type(vocabulary,tokenizer,is_destination_language,limit_to_vocab=True):
    """each  """
    class Sentence:
        """
            class that represents a sentence in a language whose vocabulary is specified by 'vocab',
            in a tokenized form
            set as attribute of class
        """
        vocab = vocabulary
        _is_destination_language = is_destination_language
        _tokenizer = tokenizer
        _limit_to_vocab = limit_to_vocab
        @classmethod
        def to_int(cls,tokens):
            """convert tokens to int using the vocabulary"""
            res = [cls.vocab[token] for token in tokens]
            return res

        @classmethod
        def from_token_int(cls,tokens_int):
            sentence_as_tokens = [cls.vocab.itos_[idx] for idx in tokens_int]
            sentence_as_tokens  = [token for token in sentence_as_tokens]
            res = cls(sentence_as_tokens) #Remark: " ".join does not reverse the tokenizer operator
            return res

        def __init__(self,sentence) -> None:
            if isinstance(sentence,str):
                _list_of_tokens = self._tokenizer(sentence)
            elif isinstance(sentence,list) and all([isinstance(el,str) for el in sentence]):
                _list_of_tokens = sentence
            else:
                raise ValueError("sentence must be a str of a list of tokens")
            # add sos and eos token is it is a destination language
            self._list_of_tokens = self._complete(_list_of_tokens)
            self._list_of_tokens_as_int = self.to_int(self._list_of_tokens)
            # show only the token recognized by the wocabulary
            self._list_of_tokens_on_vocab = [self.vocab.itos_[el] for el in self._list_of_tokens_as_int]

        @property
        def list_of_tokens_as_int(self):
            return self._list_of_tokens_as_int
        @property
        def list_of_tokens(self):
            return self._list_of_tokens

        def __len__(self):
            return len(self._list_of_tokens)
        
        def pad(self,target):
            padded_list_of_tokens = self._list_of_tokens_as_int + [self.vocab['<unk>']]* (target - len(self._list_of_tokens))
            # assert len(padded_list_of_tokens) == target + (2 if self._is_destination_language else 0)
            return padded_list_of_tokens
        
        def _complete(self,_list_of_tokens):
            if self._is_destination_language:
                completed_list_of_tokens = ['<sos>'] + _list_of_tokens + ['<eos>']
            else:
                completed_list_of_tokens = _list_of_tokens
            return completed_list_of_tokens
        def __str__(self) -> str:
            if self._limit_to_vocab:
                res = " ".join(self.as_words_from_vocab)
            else:
                res = " ".join(self.as_words)
            return res
        
        def __repr__(self) -> str:
            return str(self)

    
        @property
        def as_words(self):
            return self._list_of_tokens
        @property
        def as_words_from_vocab(self):
            return self._list_of_tokens_on_vocab
        
        @property
        def as_int(self):
            tokens_list = self._list_of_tokens_as_int
            return tokens_list
        
    return Sentence


    # def __gt__(self,other):
    #     return (self.proba > other.proba)
    # def __eq__(self,other):
    #     return self.proba == other.proba
    # def __iter__(self):
    #     return (el for el in (self._list_of_tokens,self.proba))


english_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
french_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')


path_language_info = str(dir_file.joinpath("../../models/language_info.pth"))
language_info = torch.load(path_language_info)
vocab_english = language_info["english"]["vocab"].vocab
EnglishSentence = _create_sentence_type(vocab_english,english_tokenizer,is_destination_language=False)

vocab_french = language_info["french"]["vocab"].vocab
FrenchSentence = _create_sentence_type(vocab_french,french_tokenizer,is_destination_language=True)