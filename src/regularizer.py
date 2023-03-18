### regularizes a model towards producing certain independences
from asyncio import as_completed
import torch
# from chart_parser_utils import get_masking_info, read_inputs_and_parses, get_all_hidden_states_scratch
from tqdm import tqdm
from torch.optim import Adam
# import collate
# from learnt_chart_parser_utils import get_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
from src.babylm_helpers import build_datasets_babylm
from TreeRegularizer.tree_projection import TreeProjection, get_all_hidden_states, get_pre_tokenized_info
import random

class Chart():
    def __init__(self, sim_metric, tokenizer, tree_projection):
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.tree = tree_projection
        self._cache = {}
    
    def tokenizer_helper(self, inp_slice):
        inp_dict = self.tokenizer(inp_slice, return_tensors="pt", padding=True)
        inp_tokens = inp_dict["input_ids"]
        inp_lens = inp_dict["attention_mask"].sum(dim=-1)

        return inp_tokens, inp_dict["attention_mask"], inp_lens

    def cache(self, input_str, parse_split=None):
        if input_str not in self._cache:
            sentence2idx_tuple, masked_strs, input_masks = self.tree._get_masking_info([input_str])
            slice_dict = {idx: key for idx, key in sentence2idx_tuple[0]}
            self._cache[input_str] = (slice_dict, masked_strs, input_masks)
        
        return self._cache[input_str]

    def build_scores(self, input_strs, model, start_relax_layer, tqdm_disable=True, parse_splits=None):
        device = torch.device('cuda')
        scores = [{} for _ in input_strs]
        outer_context_vecs = get_all_hidden_states(model, self.tokenizer, input_strs, tqdm_disable=True)
        outer_context_vecs = [torch.tensor(vec).to(device) for vec in outer_context_vecs]

        all_masked_strs = []
        all_input_masks = []
        all_slice_dicts = {}
        str_idx = {}

        for idx, input_str in enumerate(input_strs):
            slice_dict, masked_strs, input_masks = self.cache(input_str)
            clen = len(all_input_masks)
            for offset in range(len(masked_strs)):
                str_idx[clen + offset] = idx
                all_slice_dicts[clen + offset] = slice_dict[offset] 
                all_masked_strs.append(masked_strs[offset])
                all_input_masks.append(input_masks[offset])

        batch_size = 32 
        st = 0
    
        num_layers = model.config.n_layer
        with tqdm(total=len(all_masked_strs), disable=tqdm_disable) as progress_bar:
            while st < len(all_masked_strs):
                en = min(len(all_masked_strs),st+batch_size)
                cslice = all_masked_strs[st: en]
                inputs, len_mask, input_lens = self.tokenizer_helper(cslice)
                inputs = inputs.to(device)
                input_lens = input_lens.to(device)
                len_mask = len_mask.to(device)
                inp_len = inputs.shape[1]
                # input masks specify the inner context
                if all_input_masks is not None:
                    masks_curr = all_input_masks[st: en]
                    masks_padded = []
                    for mask in masks_curr:
                        mask_padded = mask + [0] * (inp_len - len(mask))
                        masks_padded.append(mask_padded)
                    tree_mask = torch.tensor(masks_padded).to(device)
                    # mask = self.tree.t_shaped_mask(tree_mask, len_mask, num_layers)
                    mask = [len_mask] * start_relax_layer + [tree_mask] * (num_layers - start_relax_layer)
                    mask_mult = tree_mask.unsqueeze(-1)
                else:
                    mask = [len_mask for _ in range(num_layers)]
                mask_mult = len_mask.unsqueeze(-1)
                
                layer_id = -1
                outputs = [model(inputs, attention_mask=mask, output_hidden_states=True).hidden_states[layer_id]]
                outputs = [hs * (mask_mult) for hs in outputs]

                for idx, _ in enumerate(cslice):
                    inner_vec = outputs[0][idx].sum(axis=0)
                    oidx = str_idx[idx + st]
                    i, j = all_slice_dicts[idx + st]
                    outer_vec = outer_context_vecs[oidx][0].sum(axis=0)
                    scores[oidx][(i, j)] = self.sim_metric(outer_vec, inner_vec)

                progress_bar.update(en - st)
                st = en
        return scores 


class Regularizer():
    def __init__(self, sim_metric, input_strs=None, parse_splits=None, as_hinge=False, model = None, tokenizer=None, start_relax_layer=0):
        self.sim_metric = sim_metric
        self.as_hinge = as_hinge
        self.input_strs = input_strs
        self.parse_splits = parse_splits
        self.model = model
        self.tokenizer = tokenizer
        self.start_relax_layer = start_relax_layer
        # self.preprocess(input_strs, tokenizer)
        self.tree = TreeProjection(model, tokenizer, sim_fn="cosine", normalize=True)
        self.chart = Chart(sim_metric, tokenizer, self.tree)

    def preprocess(self, input_strs, tokenizer):
        self.cache = []
        for inp_str in tqdm(input_strs):
            sentence2idx_tuple, masked_strs, input_masks = self.tree._get_masking_info([inp_str])
            self.cache.append((sentence2idx_tuple, masked_strs, input_masks))
        print("Done building cache")        

    def get_splits(self, string, chart_scores, cur_loss, split_idx):
        # Base Case:
        # If all strings are composed of only two words, return the result
        if len(string) == 1:
            device = "cuda"
            return torch.tensor(0)

        # Recursive Case:
        # Get optimum split point k
        min_k = 0
        max_sum = -float('inf')
        for k in range(len(string) - 1):
            first_score = chart_scores[(0 + split_idx, k + split_idx)]
            second_score = chart_scores[(k + split_idx + 1, len(string) + split_idx - 1)]
            current_sum = first_score + second_score
            if current_sum > max_sum:
                max_sum = current_sum
                min_k = k

        # Calculate the splits and update cur_loss
        left_loss = self.get_splits(string[:min_k + 1], chart_scores, cur_loss, 0)
        right_loss = self.get_splits(string[min_k + 1:], chart_scores, cur_loss, split_idx + min_k + 1)

        # Add the results of the recursive calls directly to cur_loss after the calls
        cur_loss = cur_loss + max_sum + left_loss + right_loss

        # Return the updated cur_loss
        return cur_loss
    
    def get_rand_splits(self, string, chart_scores, cur_loss, split_idx):
        # Base Case:
        # If all strings are composed of only two words, return the result
        if len(string) == 1:
            device = "cuda"
            return torch.tensor(0)

        # Recursive Case:
        # Get optimum split point k
        k = random.randint(0, len(string) - 2)
        first_score = chart_scores[(0 + split_idx, k + split_idx)]
        second_score = chart_scores[(k + split_idx + 1, len(string) + split_idx - 1)]
        current_sum = first_score + second_score

        # Calculate the splits and update cur_loss
        left_loss = self.get_rand_splits(string[:k + 1], chart_scores, cur_loss, 0)
        right_loss = self.get_rand_splits(string[k + 1:], chart_scores, cur_loss, split_idx + k + 1)

        # Add the results of the recursive calls directly to cur_loss after the calls
        cur_loss = cur_loss + current_sum + left_loss + right_loss

        # Return the updated cur_loss
        return cur_loss

    def __call__(self, model):
        random.seed(42)
        sampled_str_idxs = random.sample(
            range(len(self.input_strs)), k=min(len(self.input_strs), 20)
        )
        sample_strs = [self.input_strs[idx] for idx in sampled_str_idxs]

        def truncate_strings(word_list, max_words=10):
            result = []
            for string in word_list:
                words = string.split()
                truncated_string = ' '.join(words[:max_words])
                result.append(truncated_string)
            return result
    
        sample_strs = truncate_strings(sample_strs, 10)

        chart_scores_all = self.chart.build_scores(sample_strs, model, start_relax_layer=self.start_relax_layer, tqdm_disable=True)

        loss_cur = [0 for _ in range(len(sample_strs))]
        rand_loss = [0 for _ in range(len(sample_strs))]
        for idx, string in enumerate(sample_strs):
            split_string = string.split()

            # Calculate loss from tree-proj split
            loss_cur[idx] = self.get_splits(split_string, chart_scores_all[idx], cur_loss=torch.tensor(0), split_idx=0)
            loss_cur[idx] /= (2 * (len(split_string) - 1))

            # Caluculate loss from random split
            rand_loss[idx] = self.get_rand_splits(split_string, chart_scores_all[idx], cur_loss=torch.tensor(0), split_idx=0)
            rand_loss[idx] /= (2 * (len(split_string) - 1))

        lambd = 0.1
        loss = [torch.clamp(rand + lambd - cur, min=0) for cur, rand in zip(loss_cur, rand_loss)]


        avg_loss = sum(loss) / len(loss)
        return avg_loss


if __name__ == '__main__':   

    # Load Model and Tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        'exp/debug-gpt2-verysmall-babylm_10M-neg_score/debug-gpt2-verysmall-babylm_10M-run-42/checkpoint-14000'
    ) 
    tokenizer = GPT2Tokenizer.from_pretrained(
        'exp/debug-gpt2-verysmall-babylm_10M-neg_score/debug-gpt2-verysmall-babylm_10M-run-42/checkpoint-14000'
    )
    device = torch.device('cuda')
    # model.to(device)

    # Load input dataset
    input_strs = build_datasets_babylm()
    input_strs = input_strs[0:100]
  
    # Instantiate regularizer class
    regularizer = Regularizer(lambda x1, x2: F.cosine_similarity(x1, x2, dim=0), input_strs, parse_splits=None, as_hinge=True, model = model, tokenizer=tokenizer)

    loss_curr = regularizer(model)
    print(loss_curr)


    # opt = Adam(model.parameters())
    # for inp in tqdm(input_strs):
    #     scores = chart.build_scores([inp], model, 0)
    #     total_score = 0.0
    #     for key, val in scores[0].items():
    #         total_score += val
    #     total_score.backward()
    #     opt.step()
    #     model.zero_grad()
    #     # print(scores)