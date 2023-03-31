from ploomber_engine.ipython import PloomberClient
import pytest
from pathlib import Path

from translation_machine.models import transformer_mod
from translation_machine import sentence_mod
from dev import namespace_tools

# initialize client
@pytest.fixture
def train_setup(notebook_run):
    notebook_run = namespace_tools.NameSpaceAggregation(notebook_run)
    client = PloomberClient.from_path(Path("../notebooks/trainings/training_setup.ipynb"))
    train_setup = client.get_namespace(notebook_run.diffuse())
    return train_setup

@pytest.fixture
def train_data_loader(train_setup):
    train_data_loader = train_setup["train_data_loader"]
    return train_data_loader

@pytest.fixture
def model(train_setup):
    model = train_setup["model"]
    return model

@pytest.fixture
def test_dataset(train_setup):
    test_dataset = train_setup["test_dataset"]
    return test_dataset

@pytest.mark.parametrize("device",["cuda"])
def test_if_mask_is_working(model,train_data_loader,test_dataset,device):
    batch = next(iter(train_data_loader))
    batch = [el[:1] for el in batch]
    batch[:2] = [el.to(device) for el in batch[:2]]
    import pdb;pdb.set_trace()
