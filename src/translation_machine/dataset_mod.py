from torch.utils.data import Dataset
import functools
import csv,linecache
from functools import lru_cache

class DatasetFromTxt(Dataset):
    """
        class that read a list of pairs of translation from a text file, and 
    """
    def __init__(self,source_file) -> None:
        """
        we assume each sentence an it's translation, and realted comment are on the same line, and separated by a "\t" symbol,
        The first two blocks of text are respectively for the source and target language
        """
        self.source_file = source_file

    def __getitem__(self,idx):
        if not(idx < len(self)):
            raise IndexError
        res = linecache.getline(self.source_file,idx+1) # indexes begin with 1 rather 0
        res = res.split("\t")[:2]
        res = [el.replace("\u202f","") for el in res]
        return res

    @lru_cache
    def __len__(self):
        num_lines = sum(1 for _ in open(self.source_file))
        return num_lines



class SentenceDataSet:
    """
        create dataset of Sentence object from a dataset of sentences as str
    """
    def __init__(self,dataset,sentence_type_src,sentence_type_dst):
        self._dataset = dataset
        self.sentence_type_src = sentence_type_src
        self.sentence_type_dst = sentence_type_dst
    
    @functools.lru_cache
    def __getitem__(self,idx):
        sentence_src,sentence_dst = self._dataset[idx]
        res = self.sentence_type_src(sentence_src),self.sentence_type_dst(sentence_dst)
        return res
    def __len__(self):
        return len(self._dataset)