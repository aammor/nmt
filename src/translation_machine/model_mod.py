
# from torch import nn
# import torch
# class SequenceTranslator(nn.Module):
#     @classmethod
#     def load(cls,path_model):
#         tmp = torch.load(path_model)
#         model_params = tmp["model_params"]
#         model_inputs = tmp["model_inputs"]

#         sequence_translator = cls(**model_inputs)

#         sequence_translator.load_state_dict(model_params)
#         sequence_translator.model_inputs = model_inputs
#         return sequence_translator

#     def save(self,path_model):
#         state_dict_extended = {"model_params":self.state_dict(),"model_inputs":self.model_inputs}
#         torch.save(state_dict_extended,path_model)
        
#     def __init__(self,embeddings_src_size,embeddings_tgt_size,hidden_size_encoder,hidden_size_decoder,vocab_src,vocab_tgt,length_src_sentence,length_tgt_sentence,dropout=0.0):
#         super(SequenceTranslator,self).__init__()
#         self.en_embedding_layer = nn.Embedding(len(vocab_src),embeddings_src_size)
#         self.ge_embedding_layer = nn.Embedding(len(vocab_tgt),embeddings_tgt_size)

#         self.length_src_sentence = length_src_sentence
#         self.length_tgt_sentence = length_tgt_sentence

#         self.encoder = nn.LSTM(input_size=embeddings_src_size,hidden_size=hidden_size_encoder,num_layers=1,batch_first=True,dropout=dropout)
#         self.decoder = nn.LSTM(input_size=embeddings_tgt_size,hidden_size=hidden_size_decoder,num_layers=1,batch_first=True,dropout=dropout)
        

#         self.model_inputs = {
#         "embeddings_src_size":embeddings_src_size,
#         "embeddings_tgt_size":embeddings_tgt_size,
#         "hidden_size_encoder":hidden_size_encoder,
#         "hidden_size_decoder":hidden_size_decoder,
#         "vocab_src":vocab_src,
#         "vocab_tgt":vocab_tgt,
#         "length_src_sentence":length_src_sentence,
#         "length_tgt_sentence":length_tgt_sentence
#         }

#         D = 2 if self.decoder.bidirectional else 1
#         num_layers = self.decoder.num_layers
        
#         output_decoder_dim = self.decoder.hidden_size*D*num_layers
        
#         self.linear_layer = nn.Linear(output_decoder_dim,len(vocab_tgt))

#     def apply_encoder(self,en_id_tokens_batchs,en_lengths,with_all_outputs=False):
#         """_summary_

#         Args:
#             en_id_tokens_batchs (torch.Tensor): of shape (N,L) where N is the number of elements of batch and
#             L is at least greater than all the sequences.
#             It is a padded tensor representattion of N sequence whose lengths are stored on the list 'en_lengths'

#             en_lengths (List): the length of each element of the batch 

#         Returns:
#             if with_all_outputs=True, returns the encoder outputs of each element of the sequence,
#             along with the short-term state and cell state for the final element of the sequence.
#         """
#         en_embeddings = self.en_embedding_layer(en_id_tokens_batchs)
#         packed_en_embeddings = nn.utils.rnn.pack_padded_sequence(en_embeddings,en_lengths,batch_first=True,enforce_sorted=False)
#         outputs_encoder,last_states_encoder = self.encoder(packed_en_embeddings)
#         if with_all_outputs:
#             return outputs_encoder,last_states_encoder
#         else:
#             return last_states_encoder

#     def apply_decoder(self,ge_id_tokens_batchs,ge_lengths,last_states_encoder):
#         """
#             get last state of decoder as padded
#         """
#         ge_embeddings = self.ge_embedding_layer(ge_id_tokens_batchs)
#         packed_ge_embeddings = nn.utils.rnn.pack_padded_sequence(ge_embeddings,ge_lengths,batch_first=True,enforce_sorted=False)
#         out_decoder,_ = self.decoder(packed_ge_embeddings,last_states_encoder)
#         out_decoder,sequence_lengths = nn.utils.rnn.pad_packed_sequence(out_decoder,total_length=self.length_src_sentence,batch_first=True)
    
#         return out_decoder,sequence_lengths

#     def forward(self,en_id_tokens_batchs,ge_id_tokens_batchs,en_lengths,ge_lengths,last_states_encoder=None):
#         """
#             returns prediction from the next token , given the previous tokens (for german sentences) on eval mode,
#             or all the predictions.

#         Args:
#             en_id_tokens_batchs (torch.Tensor): batch of english sentence as sequence of tokenized id of shape (N,L_eng)
#             ge_id_tokens_batchs (torch.Tensor): batch of german sentence as sequence of tokenized id of shape (N,L_ger)
#             en_lengths (np.ndarray): sequence of lengths (for each english sentence) of shape (N,)
#             ge_lengths (np.ndarray): sequence of lengths (for each german sentence) of shpae (N,)
#             last_states_encoder Union[(None,[torch.Tensor,torch.Tensor])]: 
#             if None, compute the encoder from the 'english' sentences (en_id_tokens_batchs
#             +en_lengths) in form of a tuple of tensors whose shapes are respectively equal to 
#             the shapes of the decoder's state

#         Returns:
#             torch.tensor: tensor of shape (N,len_vocab) on eval mode
#               and of shape (N,L_ger,len_vocab) on train mode

#         Notations:
#             N : size of the batch
#             len_vocab : size fo the vocabulary
#             L_gen : length of longest german sentence on the dataset or the (mini)batch
#             L_eng : length of longest english sentence on the dataset or the (mini)batch

#         """
#         state_encoder_no_computed = last_states_encoder is None
#         if state_encoder_no_computed:
#             last_states_encoder = self.apply_encoder(en_id_tokens_batchs,en_lengths)
#         out_decoder,sequence_lengths = self.apply_decoder(ge_id_tokens_batchs,ge_lengths,last_states_encoder)
        
#         # apply the linear layer on :
#         outputs = {}
#         if self.training: # all the token of the output of decoder on the sequence during training
#             output = self.linear_layer(out_decoder) 
#             outputs["decoder"] = output
#             outputs["sequence_lengths"] = sequence_lengths
#         else: # only the last token of the output o the decoder 
#             out_decoder = torch.stack([a[b-1] for a,b in zip(out_decoder,sequence_lengths)])
#             output = self.linear_layer(out_decoder)
#             outputs["decoder"] = output
#         outputs["last_states_encoder"] = last_states_encoder
#         return outputs




#     @torch.no_grad()
#     def predict(self,tokens_id_english,tokens_id_german,last_states_encoder=None):
#         """
#             predict next german token id given the tokenized english sentence 
#             and the last tokens of the german sentence
#         """
#         if self.training:
#             raise ValueError("predict method is available only on eval mode")
#         tokens_id_english = torch.tensor(tokens_id_english).unsqueeze(0)
#         tokens_id_german = torch.tensor(tokens_id_german).unsqueeze(0)
#         # import pdb;pdb.set_trace()
#         en_lengths = [tokens_id_english.shape[-1]]
#         ge_lengths = [tokens_id_german.shape[-1]]
#         out = self(tokens_id_english,tokens_id_german,en_lengths,ge_lengths,last_states_encoder)
#         out_decoder_last = out["decoder"]
#         probas_tokens = nn.functional.softmax(out_decoder_last,-1)
#         best_token_id = torch.argmax(probas_tokens,dim=-1)

#         out = {}
#         out["probas"] = probas_tokens
#         out["best_token_id"] = best_token_id
#         out["last_states_encoder"] = last_states_encoder
#         return out
#     # def forward(self,en_id_tokens_batchs,ge_id_tokens_batchs,en_lengths,ge_lengths,last_states_encoder=None):


# # if not(restart):
