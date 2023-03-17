# positionnal encodings
from torch import nn
import math,torch


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
        """
           index_target_for_pred : 
           index of target to predict which implies to mask all the indexes starting from this one

        """
        assert tgt_id_tokens_batchs[0] == self.vocab_tgt_size["<sos>"] 
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

        if self.train:            
            #apply the decoder:

            out_decoder = self.transformer_decoder(embeddings_tgt,memory=encoder_output)

            out_transformer = self.linear(out_decoder)

            outputs["decoder"] = out_transformer
        else:

            out_decoder = torch.stack([a[b-1] for a,b in zip(out_transformer,sequence_lengths)])
            
            out_transformer = self.linear(out_decoder)
            
            outputs["decoder_last"] = out_transformer
        
        import pdb;pdb.set_trace()
        # outputs["decoder_last"] = 

        return res


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