# positionnal encodings
from torch import nn
import math,torch

from . import common


class TransformerForSeq2Seq(nn.Module):
    def __init__(self,d_model,vocab_src,vocab_tgt):
        super(TransformerForSeq2Seq,self).__init__()
        
        self.vocab_src_size = len(vocab_src)
        self.vocab_tgt_size = len(vocab_tgt)

        self.src_embedding_layer = nn.Embedding(self.vocab_src_size, d_model)
        self.tgt_embedding_layer = nn.Embedding(self.vocab_tgt_size, d_model)
        
        self.positionnal_encoder = PositionalEncoding(d_model)
        # self.transformer_model = nn.Transformer(d_model=d_model,batch_first=True)


        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8,batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,norm=encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8,batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6,norm=decoder_norm)

        self.linear = nn.Linear(d_model,self.vocab_tgt_size)

    def forward(self,src_id_tokens_batchs,tgt_id_tokens_batchs,src_lengths,tgt_lengths):
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            assert int(torch.unique(tgt_id_tokens_batchs[:,0])) == 1
        # the first token should always be the 'start of sequence' token whereas we are in
        # train or eval mode
        
        #compute the embeddings
        embeddings_src = self.src_embedding_layer(src_id_tokens_batchs)
        embeddings_src = self.positionnal_encoder(embeddings_src)
        
        embeddings_tgt = self.tgt_embedding_layer(tgt_id_tokens_batchs)
        embeddings_tgt = self.positionnal_encoder(embeddings_tgt)

        #apply the encoder
        encoder_output = self.transformer_encoder(embeddings_src)

        outputs = dict()

        outputs["sequence_lengths"] = tgt_lengths
        outputs["last_states_encoder"] = encoder_output

        #apply the decoder:
        max_tgt_length = embeddings_tgt.shape[1] # on the batch
        #for each tokan input (in the form of embeddings), we use (in the context of teacher-forcing) only the indexes located
        # before the token or equal to this token, thus we mask the toekns situated after the current toekn
        tgt_mask = torch.tensor([[idx_j>idx_i for idx_j in range(max_tgt_length)] for idx_i in range(max_tgt_length)],dtype=torch.bool,device="cuda")
        out_decoder = self.transformer_decoder(embeddings_tgt,memory=encoder_output,tgt_mask=tgt_mask)
    

        if self.traininig:
            # the sentence begin with sos_token and ends with eos token, 
            # the tokens from the first token to eos token excluded are given as input (their  embeddings to the transformer) 
            # the tokens from the second token to eos token included are the target tokens

            targets_tokens_for_model = [tgt_sentence_tokens[1:tgt_sentence_length]  for (tgt_sentence_tokens,tgt_sentence_length) in zip(tgt_id_tokens_batchs,tgt_lengths)]
            targets_tokens_for_model = torch.concat(targets_tokens_for_model)
    
            # import pdb;pdb.set_trace()

            predictions_lengths = [int(el)-1 for el in tgt_lengths] 
            # we remove the last token (the eos token),because, 
            # obviously, we don't need to make preidction once the sentence is over, this the offset between the length of predictions
            # and the length of each target sequence eof the batch

            # we compute the predictions, ont token at a time, until the eos token
            out_decoder = [out_decoder_instance[:length] for out_decoder_instance,length in zip(out_decoder,predictions_lengths)]
            out_decoder = torch.concat(out_decoder)
            out_transformer = self.linear(out_decoder)
            
            outputs["preds"] = out_transformer
            outputs["targets"] = targets_tokens_for_model
            with torch.no_grad():

                concatenated_tokenized_preds = torch.argmax(outputs["preds"],axis=1)

                outputs["token_preds_instance_separated"] = common.divide_into_sublist(concatenated_tokenized_preds,predictions_lengths)
                outputs["token_targets_instance_separated"] = common.divide_into_sublist(outputs["targets"],predictions_lengths)

        else:
            import pdb;pdb.set_trace()
            out_decoder = torch.stack([out_decoder_instance[length-2] for out_decoder_instance,length in zip(out_decoder,tgt_lengths)])  
            out_transformer = self.linear(out_decoder)
            
            outputs["preds"] = out_transformer


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