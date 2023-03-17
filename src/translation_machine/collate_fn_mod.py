import torch
def get_collate_fn(max_length_src,max_length_dst):
    """
        get collate function, that transform a batch of pair of 'Sentence' objects corresponding 
        to translations in two differents languages.

    """

    def collate_fn(batch):
        """transform batch of pairs of src and dst sentences into list of tensors"""
        src_id_tokens_batchs = []
        dst_id_tokens_batchs = []
        src_lengths = []
        dst_lengths = []
        for el in batch:
            src_sentence,dst_sentence = el

            src_length = len(src_sentence)
            dst_length = len(dst_sentence)

            dst_length_with_bordering_token = dst_length + 2
            assert (src_length <= max_length_src) and (dst_length <= max_length_dst)

            id_token_src = src_sentence.pad(target=max_length_src)
            id_token_dst = dst_sentence.pad(target=max_length_dst)
            
            # print(len(id_token_src),max_length_src)
            src_id_tokens_batchs.append(id_token_src)
            dst_id_tokens_batchs.append(id_token_dst)
            src_lengths.append(src_length)
            dst_lengths.append(dst_length_with_bordering_token) 

        #convert to tensors
        res =  src_id_tokens_batchs,dst_id_tokens_batchs,src_lengths,dst_lengths
        res = [torch.tensor(el) for el in res]
        
        return res
    return collate_fn




