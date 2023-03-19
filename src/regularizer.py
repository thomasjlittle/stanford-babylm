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

        batch_size = 1024 
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
                
                # # isolate score_best and score_rand
                # for idx, score in enumerate(scores):
                #     scores = list(score.values())
                #     score_best = min(scores)
                #     score_rand = random.choice(scores)
                #     scores[idx] = [score_best, score_rand]

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

    def get_splits(self, string, chart_scores, cur_loss):
        # Base Case:
        # If all strings are composed of only two words, return the result
        if len(string.split()) == 1:
            return 0

        # Recursive Case:
        # Get optimum split point k
        min_k = 0
        min_sum = float('inf')
        str_len = len(string.split())
        for k in range(str_len - 1):
            first_score = chart_scores[(0, k)]
            second_score = chart_scores(k + 1, str_len - 1)
            current_sum = first_score + second_score
            if current_sum < min_sum:
                min_sum = current_sum
                min_k = k

        # Calculate the splits and update cur_loss
        cur_loss = cur_loss + min_sum
        cur_loss = cur_loss + self.get_splits(string.split()[:min_k + 1], chart_scores, cur_loss)
        cur_loss = cur_loss + self.get_splits(string.split()[min_k + 1:], chart_scores, cur_loss)

        # Recursively call the function with the new input strings and updated cur_loss
        return cur_loss

        loss_cur = 0.0
        # for chart_score in chart_scores_all:
        #     scores = list(chart_score.values())
        #     score_best = min(scores)
        #     score_rand = random.choice(scores)

        #     lambd = 0.1
        #     loss_cur += min(score_best + lambd - score_rand, 0)

        # return loss_cur / len(idxs)

    def get_hinge(self, i, j, gold_split, inner_context_vecs, contextual_vec, keys):
        best_score = 0.0
        gold_score = self.get_score(i, j, gold_split, inner_context_vecs, contextual_vec, keys)
        for k in range(i, j):
            if k == gold_split:
                continue
            else:
                curr_score = self.get_score(i, j, k, inner_context_vecs, contextual_vec, keys)
                if curr_score > best_score:
                    best_score = curr_score
        return self._hinge_loss(best_score, gold_score)

    def _hinge_loss(self, ours, gold):
        loss = 0.1 + ours - gold
        return (loss > 0.1) * loss

    def get_score(self, i, j, k, inner_context_vecs, contextual_vec, keys):        
        cont_vec_1 = contextual_vec[i: k+1].sum(dim=0)
        inner_vec_1 = inner_context_vecs[keys[(i, k)]]

        cont_vec_2 = contextual_vec[k+1:j+1].sum(dim=0)
        inner_vec_2 = inner_context_vecs[keys[(k+1, j)]]
        return (self.sim_metric(cont_vec_1, inner_vec_1) + self.sim_metric(cont_vec_2, inner_vec_2))

    def get_hinge_2(self, i, j, k, chart_scores):
        gold_score = chart_scores[(i,k)] + chart_scores[(k+1,j)]
        return self._hinge_loss(max(chart_scores[(i,k1)] + chart_scores[(k1+1, j)] for k1 in range(i,j)), gold_score)

    def get_losses(self, chart_scores, parse_splits):
        total_loss = 0.0
        for key in parse_splits:
            i, j = key
            k = parse_splits[key]
            if self.as_hinge:
                total_loss += self.get_hinge_2(i, j, k, chart_scores)
            else:
                total_loss += -1.0*(chart_scores[(i, k)] + chart_scores[(k+1, j)])
        if len(parse_splits) > 0:
            return total_loss / len(parse_splits) # this is a loss
        else:
            return 0

    def __call__(self, model):
        random.seed(42)
        sampled_str_idxs = random.sample(
            range(len(self.input_strs)), k=min(len(self.input_strs), 6500)
        )
        sample_strs = [input_strs[idx] for idx in sampled_str_idxs]

        chart_scores_all = self.chart.build_scores(sample_strs, model, start_relax_layer=self.start_relax_layer, tqdm_disable=True)

        loss_cur = [0 for _ in range(len(sample_strs))]
        for idx, string in enumerate(sample_strs):
            loss_cur[idx] = self.get_splits(string, chart_scores_all[idx], cur_loss = torch.tensor(0).to(device))

        return loss_cur




if __name__ == '__main__':   

    # Load Model and Tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        'exp/debug-gpt2-verysmall-babylm_10M-neg_score/debug-gpt2-verysmall-babylm_10M-run-42/checkpoint-14000'
    ) 
    tokenizer = GPT2Tokenizer.from_pretrained(
        'exp/debug-gpt2-verysmall-babylm_10M-neg_score/debug-gpt2-verysmall-babylm_10M-run-42/checkpoint-14000'
    )
    device = torch.device('cuda')
    model.to(device)

    # Load input dataset
    input_strs = build_datasets_babylm()
    input_strs = input_strs[0:10]
  
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