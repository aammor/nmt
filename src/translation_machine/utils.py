def convert_to_map_dataset(dataset):
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.dataset = list(dataset)
        def __getitem__(self,idx):
            return self.dataset[idx]
        def __len__(self):
            return len(self.dataset)
    dataset = MyDataset()
    return dataset