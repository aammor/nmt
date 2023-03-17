from itertools import islice
from torch import optim
import torch
import numpy as np
from ignite.metrics.nlp import Bleu

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
        self.metric =  Bleu()


    def update_metric(self,batch_processing_out):
        preds_as_tokens = batch_processing_out["preds_as_tokens"]
        targets_as_tokens = batch_processing_out["targets_as_tokens"]
        targets_as_tokens = [[el] for el in targets_as_tokens]
        self.metric.update((preds_as_tokens,targets_as_tokens))

    def process_batch(self,batch):
        """
            compute loss and get auxiliary informations, such as number of tokens per sequence, predicted tokens
            and target tokens in the form of a returned dictionnary
        """
        try:
            batch[:2] = [el.to(self.device) for el in batch[:2]]
            # import pdb;pdb.set_trace()
            out_model = self.model(*batch)
            en_id_tokens_batchs,fr_id_tokens_batchs,en_lengths,fr_lengths = batch
            
            
            decoder_output = out_model["decoder"]
            fr_id_tokens_batch_pred_truncated = [fr_id_tokens_instance[1:length] for (fr_id_tokens_instance,length) in zip(fr_id_tokens_batchs,fr_lengths)]
            out_model_instance_pred_truncated = [out_model_instance[:length-1] for (out_model_instance,length) in zip(decoder_output,fr_lengths)]

            preds_before_softmax = torch.concat(out_model_instance_pred_truncated,axis=0)
            targets_tokens = torch.concat(fr_id_tokens_batch_pred_truncated)
            loss = self.loss_fn(preds_before_softmax,targets_tokens) #temp fix (replace by self.batch_size*max_length)
            nb_words = len(targets_tokens)


            preds_as_tokens = [[int(el) for el in  torch.argmax(el,axis=1)] for el in out_model_instance_pred_truncated]
            targets_as_tokens = [[int(el) for el in el1] for el1 in fr_id_tokens_batch_pred_truncated]
            # import pdb;pdb.set_trace()
            out = dict()
            out["loss"] = loss
            out["nb_words"] = nb_words
            out["preds_as_tokens"] = preds_as_tokens            
            out["targets_as_tokens"] = targets_as_tokens            
        except Exception as e:#exception handler that must be checked later
            print(e)
            raise LossNoneComputed
        else:
            return out 
        
    @loss_none_computed_handler
    def train_on_batch(self,batch,batch_idx):
        self.optimizer.zero_grad()
        out = self.process_batch(batch)
        loss,nb_words = out["loss"],out["nb_words"]
        # print(float(loss),batch_idx)
        loss.backward()
        self.optimizer.step()

        self.update_metric(out)
        return loss,nb_words

    @loss_none_computed_handler
    def validate_on_batch(self,batch,batch_idx):
        out = self.process_batch(batch)
        loss,nb_words = out["loss"],out["nb_words"]
        # print(float(loss),batch_idx)
        self.update_metric(out)
        return loss,nb_words

    
    def train_on_epoch(self):
        losses = []
        nb_words_per_batch = []
        self.metric.reset()
        for batch_idx,batch in enumerate(self.train_data_loader):
            loss,nb_words = self.train_on_batch(batch,batch_idx)
            if batch_idx %100 == 0:
                print(batch_idx,float(loss)/nb_words)
            if loss is not(None):
                losses.append(loss)
                nb_words_per_batch.append(nb_words)
        self.scheduler.step()
        metric_value = self.metric.compute()
        return losses,nb_words_per_batch,metric_value
    
    @torch.no_grad()
    def validate_on_epoch(self):
        losses = []
        nb_words_per_batch = []
        self.metric.reset()
        for batch_idx,batch in enumerate(self.val_data_loader):
            loss,nb_words = self.validate_on_batch(batch,batch_idx)
            if batch_idx %100 == 0:
                print(batch_idx,float(loss)/nb_words)
            if loss is not(None):
                losses.append(loss)
                nb_words_per_batch.append(nb_words)
        metric_value = self.metric.compute()
        return losses,nb_words_per_batch,metric_value
    
    

