"""module containing the definition of the model

"""
from torch import nn
import math,torch,itertools

from . import common


class TransformerForSeq2Seq(nn.Module):
    def __init__(self,d_model,vocab_src,vocab_tgt,nhead=8):
        super(TransformerForSeq2Seq,self).__init__()
        
        self.vocab_src_size = len(vocab_src)
        self.vocab_tgt_size = len(vocab_tgt)

        self.src_embedding_layer = nn.Embedding(self.vocab_src_size, d_model)
        self.tgt_embedding_layer = nn.Embedding(self.vocab_tgt_size, d_model)
        
        self.nhead = nhead
        self.positionnal_encoder = PositionalEncoding(d_model)
        # self.transformer_model = nn.Transformer(d_model=d_model,batch_first=True)


        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.nhead,batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,norm=encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=self.nhead,batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6,norm=decoder_norm)

        self.linear = nn.Linear(d_model,self.vocab_tgt_size)

    def forward(self,batch,last_states_encoder=None):
        """ the model takes as input tensor specifying as batch of pair of sentence in forms of tokens
             (src_id_tokens_batchs,tgt_id_tokens_batchs),all the sentences are represented in the form of sequence of ints 
             (each int representing the index of the tokens on the associated language), and each sentence is completed 
             with the "unknown" token.

            By source language (src),we refer the the language to be translated, and by target language (tgt)
            we refer the language into which the sentence is translated.

            For the target language, a "sos" and "eos" tokens are appended at the beginning and end of the sentence

            
        Args:
            src_id_tokens_batchs (torch.Tensor): of shape (N,src_L) reprensents the sentences of the source language
            tgt_id_tokens_batchs (torch.Tensor): of shape (N,src_T) reprensents the sentences of the target language 

            src_lengths (torch.Tensor):  the length of all the tokens used for predictions (until the padding is used)
            tgt_lengths (torch.Tensor):  the length of all the tokens used for predictions (all the tokens until the 'eos' token
            included)
            last_states_encoder (torch.Tensor, optional): value of the encoder computed from the source sentences, which can be passed
            in 'eval' mode, to avoid recomputing it.Defaults to None.

        Context:
            N is the size of the batch
            src_L is the maximmum value for the sentence's length of the srou

        Returns:
            _type_: _description_
        """
        # import pdb;pdb.set_trace()
        src_id_tokens_batchs,tgt_id_tokens_batchs,src_lengths,tgt_lengths = batch
        outputs = dict()
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            assert int(torch.unique(tgt_id_tokens_batchs[:,0])) == 1
        # the first token should always be the 'start of sequence' token whereas we are in
        # train or eval mode

        
        # compute the embeddings of the source sentence if not given, otherwise pass them directly
        # to the 'outputs' dictionnary
        if last_states_encoder is None:
            embeddings_src = self.src_embedding_layer(src_id_tokens_batchs)
            embeddings_src = self.positionnal_encoder(embeddings_src)
            #apply the encoder
            encoder_output = self.transformer_encoder(embeddings_src)

            outputs["last_states_encoder"] = encoder_output
        else:
            outputs["last_states_encoder"] = last_states_encoder


        #compute the embeddings of the target sentence
        embeddings_tgt = self.tgt_embedding_layer(tgt_id_tokens_batchs)
        embeddings_tgt = self.positionnal_encoder(embeddings_tgt)



        outputs["sequence_lengths"] = tgt_lengths

        #apply the decoder:
        max_tgt_length = tgt_id_tokens_batchs.shape[1] # on the batch
        max_src_length = src_id_tokens_batchs.shape[1] # on the batch
        batch_size = src_id_tokens_batchs.shape[0]
        #for each token input (in the form of embeddings), we use  only the indexes located
        # before the token
        tgt_mask = torch.tensor([[idx_j>idx_i for idx_j in range(max_tgt_length)] for idx_i in range(max_tgt_length)],dtype=torch.bool,device="cuda")

        memory_mask = torch.tensor([[[idx_j>src_lengths[idx_batch] for idx_j in range(max_src_length)] for _ in range(max_tgt_length)] for idx_batch in range(batch_size)],dtype=torch.bool,device="cuda")
        memory_mask = memory_mask.repeat((self.nhead,1,1))

        out_decoder = self.transformer_decoder(embeddings_tgt,memory=outputs["last_states_encoder"],tgt_mask=tgt_mask,memory_mask=memory_mask)

        # the sentence begin with sos_token and ends with eos token, 
        # the tokens from the first token to eos token excluded are given as input (their  embeddings to the transformer) 
        # the tokens from the second token to eos token included are the target tokens

        targets_tokens_for_model = [tgt_sentence_tokens[1:tgt_sentence_length]  for (tgt_sentence_tokens,tgt_sentence_length) in zip(tgt_id_tokens_batchs,tgt_lengths)]
        targets_tokens_for_model = torch.concat(targets_tokens_for_model)

        #on 'train' mode tgt_lengths include the 'eos' token, it makes sense to remove it,
        # on 'eval' mode, one token is predicted at a time and once the "eos" is predicted, the prediction is complete.
        if self.training:
            # we remove the last token (the eos token),because, 
            # obviously, we don't need to make preidction once the sentence is over, this the offset between the length of predictions
            # and the length of each target sequence eof the batch
            predictions_lengths = [int(el)-1 for el in tgt_lengths] 
        else:
            predictions_lengths = [int(el) for el in tgt_lengths]
            
        # we compute the predictions, ont token at a time, until the eos token
        out_decoder = [out_decoder_instance[:length] for out_decoder_instance,length in zip(out_decoder,predictions_lengths)]
        out_decoder = torch.concat(out_decoder)
        out_transformer = self.linear(out_decoder)

        if self.training:
     
            outputs["preds"] = out_transformer
            outputs["targets"] = targets_tokens_for_model
            with torch.no_grad():

                concatenated_tokenized_preds = torch.argmax(outputs["preds"],axis=1)
                outputs["token_preds_instance_separated"] = common.divide_into_sublist(concatenated_tokenized_preds,predictions_lengths)
                outputs["token_targets_instance_separated"] = common.divide_into_sublist(outputs["targets"],predictions_lengths)

        else:
            # import pdb;pdb.set_trace()
            outputs["preds_last"] = out_transformer[-1]

        return outputs
    
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self,tokens_list_src_int,tokens_list_tgt_int,last_states_encoder=None):
        

        src_id_tokens_batchs = torch.tensor(tokens_list_src_int).unsqueeze(0).to(self.device)
        tgt_id_tokens_batchs = torch.tensor(tokens_list_tgt_int).unsqueeze(0).to(self.device)
        
        src_lengths = torch.tensor([len(el) for el in src_id_tokens_batchs]).to(self.device).unsqueeze(0)
        tgt_lengths = torch.tensor([len(el) for el in tgt_id_tokens_batchs]).to(self.device).unsqueeze(0)
        batch = batch = [src_id_tokens_batchs,tgt_id_tokens_batchs,src_lengths,tgt_lengths]
        outputs_model = self(batch,last_states_encoder=last_states_encoder)
        predicted_token = torch.argmax(outputs_model["preds_last"])
        outputs = dict()
        outputs["next_token"] = predicted_token
        outputs["last_states_encoder"] = outputs_model["last_states_encoder"]
        return outputs

class PositionalEncoding(nn.Module):
    """with batch as first index"""

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position * div_term)
        pe[0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #import pdb;pdb.set_trace()

        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)