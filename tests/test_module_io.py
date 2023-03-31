import pytest
from dev import module_io
import requests,torch
    
@pytest.mark.parametrize("request_class",[torch.optim.Adam])
def test_if_serialization_is_reversible(request_class):
    serialization = module_io.serialize(request_class)
    class_name,module_name = serialization
    loaded_class = module_io.get_callable(class_name,module_name)
    assert loaded_class == request_class
