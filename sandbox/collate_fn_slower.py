import torch


def get_collate_fn_standard(vocab_src,vocab_dst,src_tokenizer,dst_tokenizer,max_length_src,max_length_dst):
    """
        get collate function, that transform a batch of pair of sentences corresponding 
        to translations in two differents languages.
    """
    max_length_dst_extended = max_length_dst + 2

    def collate_fn(batch):
        """transform batch of pairs of src and dst sentences into list of tensors"""
        src_id_tokens_batchs = []
        dst_id_tokens_batchs = []
        src_lengths = []
        dst_lengths = []
        for el in batch:
            src_sentence,dst_sentence = el
            id_token_src = [vocab_src[el] for el in src_tokenizer(src_sentence)]
            id_token_dst = [vocab_dst[el] for el in dst_tokenizer(dst_sentence)]
            src_length = len(id_token_src)
            dst_length = len(id_token_dst)+2
            
            id_token_src += [vocab_src['<unk>']]*(max_length_src-len(id_token_src))
            id_token_dst = [vocab_dst['<sos>']]+id_token_dst+[vocab_dst['<eos>']]
            id_token_dst += [vocab_dst['<unk>']]*(max_length_dst_extended-len(id_token_dst))
            
            #we add the start and en end of sequence token to each spanish sentence
            src_id_tokens_batchs.append(id_token_src)
            dst_id_tokens_batchs.append(id_token_dst)
            src_lengths.append(src_length)
            dst_lengths.append(dst_length)

        #convert to tensors
        res =  src_id_tokens_batchs,dst_id_tokens_batchs,src_lengths,dst_lengths
        # import pdb;pdb.set_trace()
        res = [torch.tensor(el) for el in res]
        
        return res
    return collate_fn


