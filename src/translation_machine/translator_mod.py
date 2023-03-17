import torch,numpy as np
from translation_machine.sentence_mod import FrenchSentence

DstSentence = FrenchSentence

class Translator:
    """src to german translator using a trained NLP model"""

    def __init__(self,model,beam_search_width=3):
        self.beam_search_width = beam_search_width
        self.model = model
        self.model.eval()

    
    def __call__(self,src_sentence,limit_sentence = np.inf):
        tokens_list_src_int = src_sentence.as_int
        tokens_list_dst_int = [DstSentence.vocab['<sos>']]

        current_id_dst = None
        predictions = {"last_states_encoder":None} # to be given as arguments to the predict method
        while (current_id_dst != DstSentence.vocab['<eos>'] and len(tokens_list_dst_int)<240 ):
            predictions = self.model.predict(tokens_list_src_int,tokens_list_dst_int,last_states_encoder=predictions["last_states_encoder"])
            probs = predictions["probas"]
            current_id_dst = int(torch.argsort(probs,descending=True).squeeze()[0]) # get more probable token from the list
            tokens_list_dst_int.append(current_id_dst)# print(current_id_dst)

            if len(tokens_list_dst_int)>limit_sentence:
                break
        
        sentence = DstSentence.from_token_int(tokens_list_dst_int)
        return sentence

    # @torch.no_grad()
    # def compute_with_beam_search(self,src_sentence,beam_width=3,limit_sentence = np.inf,show_all=False):
    #     tokens_list_src_int = src_sentence.as_int
    #     tokens_list_dst_int = [DstSentence.vocab['<sos>']]

    #     current_id_dst = None
    #     predictions = {"last_states_encoder":None} # to be given as arguments to the predict method
    #     while (current_id_dst != DstSentence.vocab['<eos>'] ):
    #         predictions = self.model.predict(tokens_list_src_int,tokens_list_dst_int,last_states_encoder=predictions["last_states_encoder"])
    #         probs = predictions["probas"]
    #         # import pdb;pdb.set_trace()
    #         current_id_dst = torch.argsort(probs,descending=True).squeeze().numpy[:beam_width] # get more probable token from the list
    #         # print(current_id_dst)
    #         tokens_list_dst_int.append(current_id_dst)
    #         if len(tokens_list_dst_int)>limit_sentence:
    #             break
    #     sentence = DstSentence(tokens_list_dst_int)
    #     return sentence

    # def get_k_best_candidates(self,best_sentences_with_prob,last_states_encoder):
    #     """starting from k sentences , find the tokens we can add from the vocabulary 
    #         that gives k sentences with the highest probability
    #     """
    #     probs_for_best_sentences = []
    #     for sentence_with_prob in best_sentences_with_prob:
    #         probs = self.get_next_token_probabilities(last_states_encoder,sentence_with_prob)
    #         probs_for_best_sentences.append(probs)
    #     probs_for_best_sentences = torch.stack(probs_for_best_sentences)
        
    #     indexes = np.argsort(probs_for_best_sentences.detach().numpy().flatten())[::-1][:self.beam_search_width] 
    #     indices = np.unravel_index(indexes,shape=(probs_for_best_sentences.shape))
    #     sentences_indexes = indices[0]
    #     tokens_indexes = indices[1]
    #     probs_best_candidates = [probs_for_best_sentences[idx_sentence,idx_token]  
    #                             for (idx_sentence,idx_token) in zip(sentences_indexes,tokens_indexes)]

    #     new_best_sentence_with_prob = []
    #     for cnt,(idx_sentence,idx_token) in enumerate(zip(sentences_indexes,tokens_indexes)):
    #         old_sentence,prob_old_sentence = best_sentences_with_prob[idx_sentence]
    #         new_sentence = old_sentence+[int(idx_token)]
    #         new_prob = prob_old_sentence*probs_best_candidates[cnt]
    #         new_best_sentence_with_prob.append(DstSentence(new_sentence,new_prob))        
    #     return new_best_sentence_with_prob
    
    
    # @torch.no_grad()
    # def compute_with_beam_search(self,src_sentence,max_iter=5,show_all=False):
    #     last_states_encoder = self.get_encoder_last_state(src_sentence)
    #     best_sentences = [DstSentence([DstSentence.vocab['<sos>']],1) for _ in range(1)]

    #     for _ in range(max_iter):
    #         incomplete_best_sentences = [sentence_with_prob for sentence_with_prob in best_sentences
    #                                               if not(sentence_with_prob.is_complete())]
    #         complete_best_sentences = [sentence_with_prob for sentence_with_prob in best_sentences 
    #                                               if sentence_with_prob.is_complete()]
    #         if len(incomplete_best_sentences)==0:
    #             break
    #         tmp =  self.get_k_best_candidates(incomplete_best_sentences,last_states_encoder)
    #         best_sentences = sorted([*tmp,*complete_best_sentences],reverse=True)
    #         best_sentences = list(best_sentences)[:self.beam_search_width]

    #     if show_all:
    #         return best_sentences
    #     else:
    #         res = best_sentences[0]
    #         return res