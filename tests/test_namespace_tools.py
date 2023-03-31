import pytest
from argparse import Namespace
from functools import partial
import tempfile
import torch
from pathlib import Path

from dev import namespace_tools

@pytest.fixture(scope="module")
def nested_namespace():
    nested_namespace = Namespace(
    ab = Namespace(
        a = 1,
        b = 2
    ),
    # parameters to limit the size of the dataset
    cd = Namespace(
        c= 1,
        d = partial(torch.optim.NAdam,lr=0.001,momentum=0.9),
        e = True,
        f = "path",
        g = Namespace(h=1,i=10)
    ))
    return nested_namespace

@pytest.fixture()
def namespace_aggregation(nested_namespace):
    nested_namespace_aggregation = namespace_tools.NameSpaceAggregation(nested_namespace)
    return nested_namespace_aggregation

@pytest.fixture()
def state_dict(namespace_aggregation):
    state_dict = namespace_aggregation.state_dict()
    return state_dict



def test_if_state_dict_reversible(namespace_aggregation):
    state_dict = namespace_aggregation.state_dict()
    new_namespace_aggregation = namespace_tools.NameSpaceAggregation.load_state_dict(state_dict)
    assert new_namespace_aggregation == namespace_aggregation

def test_if_state_dict_is_serialized(state_dict):
    file = tempfile.NamedTemporaryFile()
    torch.save(state_dict,file.name)
    new_state_dict = torch.load(file.name)
    assert state_dict == new_state_dict

@pytest.fixture()
def non_existing_file_names():
    nb = 10
    # with tempfile.NamedTemporaryFile() as temporaryfile:
    temporary_files = [tempfile.NamedTemporaryFile() for _ in range(nb)]
    non_existing_filenames = [file.name for file in temporary_files]
    return non_existing_filenames # files exists only within scope of the function

@pytest.fixture()
def existing_files_with_root_directory():
    nb = 10
    tmpdir = tempfile.TemporaryDirectory()
    tempfiles = [tempfile.NamedTemporaryFile(dir=tmpdir.name) for _ in range(nb)]
    return tmpdir,tempfiles


def test_if_Paths_verify_integrity_of_paths(existing_files_with_root_directory,non_existing_file_names):
    tmpdir,existing_filenames = existing_files_with_root_directory
    existing_filesnames_sa_dict = {str(idx).zfill(3):file.name for idx,file in enumerate(existing_filenames)}
    tmp = {str(idx).zfill(3):filename for idx,filename in enumerate(non_existing_file_names)}

    paths1 = namespace_tools.Paths(root=str(tmpdir.name),**existing_filesnames_sa_dict)
    with pytest.raises(AssertionError):
        paths2 = namespace_tools.Paths(root=str(tmpdir.name),**tmp)

    with pytest.raises(AssertionError):
        paths3 = namespace_tools.Paths(**existing_filesnames_sa_dict)