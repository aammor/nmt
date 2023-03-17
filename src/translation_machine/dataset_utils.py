import torchtext,torch,numpy as np
from itertools import islice

class DatasetSlicer(torch.utils.data.Dataset):
    def __init__(self,dataset,start_index,stop_index):
        self.start_index = start_index
        self.stop_index = stop_index
        self.dataset = list(islice(dataset,start_index,stop_index))
    def __getitem__(self,index):
        res = self.dataset[index]
        return res
    def __len__(self):
        return self.stop_index-self.start_index

# m = 3
# len_trn = len(list(train_dataset))
# if m>0:
#     train_dataset = torch.utils.data.Subset(convert_to_map_dataset(train_dataset),np.arange(len_trn-m,len_trn))
#     val_dataset = torch.utils.data.Subset(convert_to_map_dataset(val_dataset),np.arange(m))
# else:
#     train_dataset = convert_to_map_dataset(train_dataset)
#     val_dataset = convert_to_map_dataset(val_dataset)
