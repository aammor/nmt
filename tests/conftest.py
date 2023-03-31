import torch,inspect,argparse,pytest
import numpy as np
from pathlib import Path
from functools import partial
from argparse import Namespace

from dev import namespace_tools


# nested namespace arguement containing all elements associated to the training setup

@pytest.fixture()
def notebook_run():
    notebook_run = Namespace(
        simple_hp = Namespace(
            batch_size= 16,
            d_model = 64,
            early_stop_thresh = np.inf,
            nb_epochs = 100,
            warm_up_epochs = 20,
        ),
        # parameters to limit the size of the dataset
        dset_truncation = Namespace(
            limit_length= 32,
            use_splitting = False,
            max_length_from_file = False,
        ),
        # parameters for the optimization algorithm
        opt_params = Namespace(
            unlinked_optimizer = partial(torch.optim.AdamW,lr=0.001),
            unlinked_scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', 
                                        factor=0.5, patience=15,min_lr=10**(-6))
        ),
        # parameters to reload the model
        train_state_control = Namespace(             
            load_from_backup = True,
            restore_optimizer = True
        ),
        #paths from root
        paths = namespace_tools.Paths(
            path_dataset = "data/french_english_dataset/fra.txt",
            path_language_info = "models/language_info.pth",
            path_dataset_splitting = "dataset_splitting",
            path_model_and_dependencies = f"models/sequence_translator_transformer_over_fitted_next.pth",
            root = "../"
        )
        
    )
    return notebook_run