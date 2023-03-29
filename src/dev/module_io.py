"""methods to serialize a module (convert it into a set of strings that could be saved 
        and converted back to the same module
"""
import inspect,importlib

def _get_module_name(callable):
    """name of the module where the callable is defined"""
    res = inspect.getmodule(callable)
    name_module = res.__name__
    return name_module

def serialize(callable):
    """
        convert callable into a set of strings, that are sufficient to call it back,
        namely in this case, it is the name of the callable and the module where it is defined
    """
    res =  (callable.__name__,_get_module_name(callable))
    return res


def get_callable(callable_name,module_name):
    """from the name of the class and the module name, dynamically import the callable

    Args:
        callable_name (str): the name of the class (it doesn't contain) the module that contains the class
        module_name (str): the name of the module the contains the class

    Returns:
        the class pointed by the callable_name
    """
    assert isinstance(callable_name,str)
    assert isinstance(module_name,str)
    module = importlib.import_module(module_name)
    name_to_class = dict(inspect.getmembers(module,inspect.isclass))
    if callable_name in name_to_class.keys():
        result = name_to_class[callable_name]
        return result
    else:
        raise  ValueError(f"no class with name {callable_name} is contained on module  {module_name}]")