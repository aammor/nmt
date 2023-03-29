from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from dev import module_io 


@dataclass
class NameSpaceAggregation:
    def __init__(self,namespace,) -> None:
        """_summary_

        Args:
            namespace (Iterable[NameSpace]): list of namespaces to hold
        """
        assert type(namespace) == Namespace
        self._namespace = namespace
        #set the attribute of the namespace as attribute of the current class
        self.__dict__.update(**self._namespace.__dict__)
        #make them globally accessible (but is it a good idea?)

    def diffuse(self,the_dict=None):
        """
            add the namespace to the dictionnary, 'the_dict'
            it can be used on the globals() dictionnary as input or on a 
        """
        if the_dict is None:
            return self._namespace.__dict__
        else:
            the_dict.update(**self._namespace.__dict__)
    
    def state_dict(self):
        state_dict = self._convert_to_nested_state_dict(self._namespace)
        # import pdb;pdb.set_trace()
        return state_dict
    @classmethod
    def load_state_dict(cls,state_dict):
        self = cls(cls._convert_to_nested_namespace(state_dict))
        return self
    @classmethod
    def _convert_to_nested_state_dict(cls,el):
        if type(el) ==  Namespace:
            state_dict = dict()
            for key,val in el.__dict__.items():
                state_dict[key] = cls._convert_to_nested_state_dict(val)
        elif type(el) == Paths:
                state_dict = el.as_dict
        elif type(el) == partial:
            state_dict = {"func":module_io.serialize(el.func),
            "args":el.args,
            "keywords":el.keywords
            }
        elif isinstance(el,(int,float,bool,str)):
            state_dict =  el
        else:
            raise ValueError(f"type of argument {el} of argument cannot be saved for the moment")
        return state_dict
    
    @classmethod
    def _convert_to_nested_namespace(cls,el):
        if type(el) ==  dict:
            if set(el.keys()) == {'func','args','keywords'}:
                func_as_strs = el['func']
                args = el['args']
                keywords = el['keywords']
                assert (isinstance(func_as_strs,tuple) 
                        and all([isinstance(el,str) for el in func_as_strs])
                        ),'func key must point to a function'
                assert isinstance(args,tuple),'args must points to a tuple of (positionnal) arguments'
                assert isinstance(keywords,dict),'keywords must points to a dictionnary of keywords arguments'
                func = module_io.get_callable(*func_as_strs)
                namespace = partial(func,*args,**keywords)     
            else:
                namespace = Namespace(**{key:cls._convert_to_nested_namespace(val) 
                            for (key,val) in el.items()})
        elif isinstance(el,(int,float,bool,str)):
            namespace =  el
        else:
            raise ValueError(f"type of argumen {el} of argument cannot be converted to namespace for the moment")
        return namespace      

    

class Paths(Namespace):
    def __init__(self,**kwargs):
        self.as_dict = kwargs.copy()
        assert "root" in kwargs,"root must be given with the paths"
        self.as_dict["root"] = str(Path(self.as_dict["root"]).resolve())

        self.root = kwargs.pop("root")
        kwargs = {key:Path(self.root).joinpath(val).absolute() for key,val in kwargs.items()}
        for _,val in kwargs.items():
            assert val.exists(),f"path {val} doesn't exists"
        kwargs = {key:str(val) for (key,val) in kwargs.items()}
        super(Paths,self).__init__(**kwargs)

# paths = Paths(
#     root = "../..",
#     path_dataset = "data/french_english_dataset/fra.txt",
#     path_language_info = "models/language_info.pth",
#     path_dataset_splitting = "dataset_splitting",
#     path_model_and_dependencies = "models/sequence_translator_transformer_new.pth"
# )