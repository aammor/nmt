from itertools import islice
from torch import optim
import torch,itertools
import numpy as np
# from ignite.metrics.nlp import Bleu
import time

#from ignite.metrics.nlp import Bleu



class LossNoneComputed(Exception):
    """Raised when loss couldn't be computed"""

def loss_none_computed_handler(func):
    def inner_function(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except LossNoneComputed as e:
            raise e
            print(f"couldn't  apply the model on the current batch  ")# we add exception hanndlers later
            return None
        else:
            return res
    return inner_function


class ModelTrainer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self,model,optimizer,scheduler,train_data_loader,val_data_loader,loss_fn):
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.loss_fn = loss_fn
        # self.metric =  Bleu()


    def update_metric(self,batch_processing_out):
        preds_tokens = batch_processing_out["preds_tokens"]
        targets_tokens = batch_processing_out["targets_tokens"]
        targets_tokens = [[el] for el in targets_tokens]
        # import pdb;pdb.set_trace()
        self.metric.update((preds_tokens,targets_tokens))

    def process_batch(self,batch):
        """
            compute loss and get auxiliary informations, such as number of tokens per sequence, predicted tokens
            and target tokens in the form of a returned dictionnary
        """
        try:
            batch[:2] = [el.to(self.device) for el in batch[:2]]
            out_model = self.model(batch)
            

            # import pdb;pdb.set_trace()
            preds = out_model["preds"]
            targets_tokens = out_model["targets"]
            loss = self.loss_fn(preds,targets_tokens) #temp fix (replace by self.batch_size*max_length)
            nb_words = len(targets_tokens)
            # import pdb;pdb.set_trace()
            out = dict()
            out["loss"] = loss
            out["nb_words"] = nb_words
            out["preds_tokens"] = out_model["token_preds_instance_separated"]          
            out["targets_tokens"] = out_model["token_targets_instance_separated"]
        except Exception as e:#exception handler that must be checked later
            print(e)
            raise LossNoneComputed
        else:
            return out 
        
    @loss_none_computed_handler
    def train_on_batch(self,batch,batch_idx=None):
        self.optimizer.zero_grad()
        out = self.process_batch(batch)
        loss,nb_words = out["loss"],out["nb_words"]
        # print(float(loss),batch_idx)
        loss.backward()
        self.optimizer.step()

        # self.update_metric(out)
        return loss,nb_words

    @loss_none_computed_handler
    def validate_on_batch(self,batch,batch_idx=None):
        out = self.process_batch(batch)
        loss,nb_words = out["loss"],out["nb_words"]
        # print(float(loss),batch_idx)
        # self.update_metric(out)
        return loss,nb_words

    
    def train_on_epoch(self):
        losses = []
        nb_words_per_batch = []
        # self.metric.reset()
        for batch_idx,batch in enumerate(self.train_data_loader):
            loss,nb_words = self.train_on_batch(batch,batch_idx)
            if batch_idx %100 == 0:
                print(batch_idx,float(loss)/nb_words)
            if loss is not(None):
                losses.append(loss)
                nb_words_per_batch.append(nb_words)
        self.scheduler.step()
        # metric_value = self.metric.compute()
        metric_value = None
        return losses,nb_words_per_batch,metric_value
    
    @torch.no_grad()
    def validate_on_epoch(self):
        losses = []
        nb_words_per_batch = []
        # self.metric.reset()
        for batch_idx,batch in enumerate(self.val_data_loader):
            loss,nb_words = self.validate_on_batch(batch,batch_idx)
            if batch_idx %100 == 0:
                print(batch_idx,float(loss)/nb_words)
            if loss is not(None):
                losses.append(loss)
                nb_words_per_batch.append(nb_words)
        # metric_value = self.metric.compute()
        metric_value = None
        return losses,nb_words_per_batch,metric_value
    
    def reset_optimizer(self):
        self.optimizer = self.optimizer.__class__(self.model.parameters(), **self.optimizer.defaults)
          

    def set_learning_rate(self,lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
            
    def lr_range_test(self,min_lr_search,max_lr_search,size_experiments_step,cycle_schedule,method):
        
        total_iters_schedule = size_experiments_step//cycle_schedule

        self.reset_optimizer()

        if method=="exponential":
            self.set_learning_rate(min_lr_search)
            gamma = np.exp(np.log(max_lr_search/min_lr_search)/total_iters_schedule)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=gamma)
        elif method == "linear":
            self.set_learning_rate(max_lr_search)
            start_factor = max_lr_search/min_lr_search
            scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,start_factor=start_factor,end_factor=1.0,total_iters = total_iters_schedule)
        else:
            raise ValueError

        # iterate over the train_data_loader, while shuffling at each epoch , for a certain number of iteratiosn
        cyclic_shuffled_train_data_loader = (el  for _ in itertools.count(start=0,step=1) for el in train_data_loader)
        cyclic_shuffled_train_data_loader = itertools.islice(cyclic_shuffled_train_data_loader,size_experiments_step)


        losses_lr = []
        lrs = []

        for idx,batch in enumerate(cyclic_shuffled_train_data_loader):
            print(idx)
            loss,nb_words = model_trainer.train_on_batch(batch)
            loss = float(loss)
            if idx%cycle_schedule==0:
                print(idx,scheduler.get_last_lr())
                scheduler.step()
            print(loss/nb_words,scheduler.get_last_lr())
            losses_lr.append(loss/nb_words)
            lrs.append(scheduler.get_last_lr())
        return losses_lr,lrs
