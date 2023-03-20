"""
    function usefull for the common models
"""
import torch

def divide_into_sublist(concatenated_tensor,lengths,to_numpy=True):
    stops = torch.cumsum(torch.IntTensor([0]+lengths),0)
    res = [concatenated_tensor[stop0:stop1] for (stop0,stop1) in zip(stops[:-1],stops[1:])] 
    if to_numpy:
        res = [list(el.cpu().numpy()) for el in res]  
    return res