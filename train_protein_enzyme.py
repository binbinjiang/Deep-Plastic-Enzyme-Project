import argparse
from argparse import Namespace
import numpy as np

import torch.nn.functional as F
import torch
import random
import torch.nn as nn
import os
from util.load_datasets import load_train_dataset, load_eval_dataset, load_plastic_dataset

import itertools
import string
from typing import List, Tuple
from Bio import SeqIO
from scipy.spatial.distance import squareform, pdist, cdist

from util.contact_metric import evaluate_prediction
from esm_local.modules import ContactPredictionHead

from util.kldiv_loss import SeqKD

import math


def eval_rosetta_raw_esm(esm_seq_model, esm_seq_batch_converter, eval_dataset, device):
    sum_pl, sum_pl2, sum_pl5 = 0., 0., 0.
    for num, eval_bt in enumerate(eval_dataset):
        contact_ref, seq = eval_bt
        # print(len(seq), contact_ref.shape)
        # predict_contact = torch.tensor(contact_ref).bool()
        seq_data = [('', seq)]
        seq_batch_labels, seq_batch_strs, seq_batch_tokens = esm_seq_batch_converter(seq_data)
        seq_batch_tokens = seq_batch_tokens.to(device)

        with torch.no_grad():
            seq_results = esm_seq_model(seq_batch_tokens, repr_layers=[33], return_contacts=True)
            predict_contact = seq_results["contacts"]

        metrics = evaluate_prediction(predict_contact, contact_ref)
        sum_pl += metrics["long_P@L"]
        sum_pl2 += metrics["long_P@L2"]
        sum_pl5 += metrics["long_P@L5"]

        avg_pl, avg_pl2, avg_pl5 = sum_pl/(num+1), sum_pl2/(num+1), sum_pl5/(num+1)

        if (num+1)%500==0:
            print(f"{num+1} -> P@L:{avg_pl}, P@L2:{avg_pl2}, P@L5:{avg_pl5}")
    print(f"TOTAL:{num+1} -> P@L:{avg_pl}, P@L2:{avg_pl2}, P@L5:{avg_pl5}")

    return avg_pl, avg_pl2, avg_pl5


def eval_rosetta_on_fly(esm_seq_model, esm_seq_batch_converter, eval_dataset, device): 
    sum_pl, sum_pl2, sum_pl5 = 0., 0., 0.
    valid_num = 0 
    
    mseloss= nn.MSELoss()
    # loss_ce = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(esm_seq_model.contact_head.parameters())

    for num, eval_bt in enumerate(eval_dataset):
        contact_ref, seq = eval_bt
        # print(len(seq), contact_ref.shape)
        # predict_contact = torch.tensor(contact_ref).bool()
        seq_data = [('', seq)]
        seq_batch_labels, seq_batch_strs, seq_batch_tokens = esm_seq_batch_converter(seq_data)
        seq_batch_tokens = seq_batch_tokens.to(device)

        if num<20: # for training contact predictor
            seq_results = esm_seq_model(seq_batch_tokens, repr_layers=[33], return_contacts=True)
            # seq_results["attentions"] = seq_results["attentions"].detach()  # TODO is it important here
            predict_contact = seq_results["contacts"]

            contact_ref_format = torch.tensor(contact_ref).unsqueeze(0).to(device=device, dtype=torch.float32)
            contact_loss = mseloss(contact_ref_format, predict_contact)
            
            optimizer.zero_grad()
            contact_loss.backward()
            optimizer.step()
        
        else:
            with torch.no_grad():
                seq_results = esm_seq_model(seq_batch_tokens, repr_layers=[33], return_contacts=True)
            predict_contact = seq_results["contacts"]

            metrics = evaluate_prediction(predict_contact, contact_ref)
            sum_pl += metrics["long_P@L"]
            sum_pl2 += metrics["long_P@L2"]
            sum_pl5 += metrics["long_P@L5"]

            valid_num += 1

            avg_pl, avg_pl2, avg_pl5 = sum_pl/(valid_num), sum_pl2/(valid_num), sum_pl5/(valid_num)

    return avg_pl, avg_pl2, avg_pl5


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, fixed_len, output_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.out0 = nn.Linear(embed_dim, 1)

        self.out1 = nn.Linear(fixed_len, output_dim, bias=True)

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.out0(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out1(x)
        return x


class EnzymeModel(nn.Module):
    def __init__(
        self,
        esm_seq_model,
        embed_dim,
        fixed_len
    ):
        super().__init__()
        self.esm_seq_model = esm_seq_model
        
        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            fixed_len=fixed_len,
            output_dim=2
        )

    def forward(
        self,
        seq_batch_tokens,
        repr_layers=[33], 
        return_contacts=False
    ):
        seq_results = self.esm_seq_model(seq_batch_tokens, repr_layers=[33], return_contacts=False)
        # raw_seq_contacts = seq_results["contacts"]
        token_representations = seq_results["representations"][33]
        # print(token_representations.shape)
        bi_class = self.lm_head(token_representations)

        return seq_results, bi_class

    def infer_class(
        self,
        seq_batch_tokens,
        repr_layers=[33], 
        return_contacts=False,
        target_class=None
    ):
        L = seq_batch_tokens.size(0)
        class_pred_arr = []
        bts = L//8 if L%8==0 else L//8+1

        for k in range(bts):
            seq_batch_tokens_split = seq_batch_tokens[k*8:(k+1)*8]
            with torch.no_grad():
                seq_results = self.esm_seq_model(seq_batch_tokens_split, repr_layers=[33], return_contacts=False)
            token_representations = seq_results["representations"][33]
            bi_class = self.lm_head(token_representations)

            class_pred = torch.argmax(torch.softmax(bi_class, dim=-1), dim=-1)
            class_pred_arr += class_pred.cpu().numpy().tolist()

        class_pred_tensor = torch.tensor(class_pred_arr).to(target_class.device)
        acc = sum(class_pred_tensor==target_class).item()/L
        try:
            pos_acc = sum((class_pred_tensor==target_class)*target_class).item()/sum(target_class).item()
            # print(f"pos_acc={pos_acc}")
        except:
            pos_acc=-1
            # print(f"pos_acc=None")
        try:
            neg_acc = sum((class_pred_tensor==target_class)*(1-target_class)).item()/sum(1-target_class).item()
            # print(f"neg_acc={neg_acc}")
        except:
            neg_acc=-1
            # print(f"neg_acc=None")

        return acc, pos_acc, neg_acc

def train_esm_enzyme(Enzyme_Model, seq_alphabet, device, args):

    kl_align_loss = SeqKD(T=1)
    contact_mse_loss = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss(reduction="none")
    loss_ce_mean = nn.CrossEntropyLoss(reduction="mean")

    seq_batch_converter = seq_alphabet.get_batch_converter()
    print(seq_alphabet.to_dict())
    alphabet_all_toks = seq_alphabet.all_toks

    print(f"args.optimizer_type.lower()=={args.optimizer_type.lower()}")
    if args.optimizer_type.lower()=="sgd":
        # default: lr=1e-3, weight_decay=0
        optimizer = torch.optim.SGD(Enzyme_Model.parameters(), lr=args.lr) 
    elif args.optimizer_type.lower()=="adamw":
        # default: lr=1e-3 betas=(0.9, 0.999), weight_decay=0.01
        optimizer = torch.optim.AdamW(Enzyme_Model.parameters(), lr=args.lr)
    elif args.optimizer_type.lower()=="adam":
         # default: lr=1e-3 betas=(0.9, 0.999), weight_decay=0
        optimizer = torch.optim.Adam(Enzyme_Model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)
    

    if args.train_data_type=="debug":
        # batch_list = ['DREVLSKENQFVVQVPKSCSLDIKTQIRSALKAAITSEQ', 'NRRQHRLINLSIFGRQVGLRFADAVAVADAGNPVGSRGKRATGTEEADPQTIREQAALDAA', 'LSWEPPKYDGGSSINNYIVEKRDTSTTAWQIVSATVARTTIKACRLKTGCEYQFRIAAEN', 'CKVKVTVRERPGALPPAAEVPAAAPTKKLDNVFFIEEPKPAHVTEKGTATFIAKVGGDPI', 'VHDPRVLDEDKAIKQFVPLSDMKWYRKLRDQYEIPERMERILQKRQRRIRLSRWEQFYVM']
        uniref_batch_list_all = load_train_dataset(split_name="debug")
    elif args.train_data_type=="uniref50":
        uniref_batch_list_all = load_train_dataset(split_name="uniref50")
    else:
        raise("wrong args.train_data_type")

    PlasticDB_batch_list_all = load_plastic_dataset(split_name="PlasticDB")
    random.shuffle(PlasticDB_batch_list_all)

    uniref50_len = 5000
    PlasticDB_batch_list = PlasticDB_batch_list_all[:50] + uniref_batch_list_all[:uniref50_len]

    uniref_batch_list = uniref_batch_list_all[uniref50_len:]

    PlasticDB_batch_list_train = PlasticDB_batch_list_all[50:]

    if args.eval_data_type=="debug":
        eval_batch_list = load_eval_dataset(split_name="debug")
    elif args.train_data_type=="rosetta_dataset":
        eval_batch_list = load_eval_dataset(split_name="rosetta_dataset")
    elif args.eval_data_type=="eval_for_train":
        eval_batch_list = load_eval_dataset(split_name="eval_for_train")
    else:
        raise("wrong args.eval_data_type")

    bt = args.batch_size
    epochs = args.epochs

    Enzyme_Model = Enzyme_Model.train()

    mlm_prob = 0.2
    MASK_token = 32 # '<mask>': 32

    esm_model = Enzyme_Model.esm_seq_model
    for ep in range(epochs):
        random.shuffle(uniref_batch_list)

        batch_list = PlasticDB_batch_list_train + uniref_batch_list[:1500]

        batch_list = batch_list[:len(batch_list)-len(batch_list)%bt]
        assert len(batch_list)%bt == 0

        random.shuffle(batch_list)

        sum_loss = 0.
        sum_loss_class = 0.
        sum_loss_mask = 0.

        i = 0
        bts = len(batch_list)//bt
        for _ in range(bts):
            # batch = [seq1, seq2, ..., seqn]
            batch_raw = batch_list[i*bt:bt*(i+1)]
            # print(batch)

            fixed_len = args.fixed_len # 256
            max_len= fixed_len


            crop_true_len = [min(len(seq), max_len) for clas_id, seq in batch_raw]
            target_class = torch.tensor([clas_id for clas_id, seq in batch_raw]).to(device)

            seq_data = [(str(idx),seq[:crop_true_len[idx]]) for idx, (class_id,seq) in enumerate(batch_raw)]
            
            # print(seq_data)
            # print(crop_true_len)
            
            seq_batch_labels, seq_batch_strs, seq_batch_tokens_tmp = seq_batch_converter(seq_data)
            
            # print(seq_batch_tokens_tmp)
            # padding seq_batch_tokens to a fixed length=256 for classfication
            seq_batch_tokens = torch.ones(seq_batch_tokens_tmp.size(0), fixed_len+2) # '<pad>': 1

            bt_max_len = max(crop_true_len)+2
            assert bt_max_len==seq_batch_tokens_tmp.size(1)
            seq_batch_tokens[:,:bt_max_len] = seq_batch_tokens_tmp

            seq_batch_tokens = seq_batch_tokens.to(device=device, dtype=seq_batch_tokens_tmp.dtype)

            ori_seq_batch_tokens = seq_batch_tokens.clone().detach()

            B = len(crop_true_len)
            max_L= max(crop_true_len)

            true_len_torch = torch.tensor(crop_true_len)
            
            seq_range = torch.arange(start=0, end=max_L, step=1)
            seq_range_expand  = seq_range.unsqueeze(0).expand(B, max_L)
            # print(seq_range_expand)
            seq_length_expand = true_len_torch.unsqueeze(-1).expand_as(seq_range_expand)
            # print(seq_length_expand)
            seq_mask_tmp  = seq_range_expand < seq_length_expand
            seq_mask = torch.zeros(B, fixed_len+2)
            seq_mask[:,1:1+max_L] =  seq_mask_tmp
            seq_mask = seq_mask.to(device)
            
            # assert seq_batch_tokens.size()[-1]<=1024-2
            # L_max = seq_batch_tokens.size()[-1]-2
            # print("L_max:",seq_batch_tokens.size(), L_max)

            # mlm_prob denotes the probability of number 1
            out_mask = torch.zeros(B, fixed_len+2)
            for b in range(B):
                curr_len = crop_true_len[b]
                out_mask_tmp = torch.zeros(1, curr_len+2)
                out_mask_tmp[:,1:1+curr_len] = torch.bernoulli(torch.full([1, curr_len], mlm_prob))
                out_mask_tmp = out_mask_tmp.to(device)
                seq_batch_tokens[b,:curr_len+2] = (1-out_mask_tmp)*seq_batch_tokens[b,:curr_len+2] + out_mask_tmp * MASK_token
                out_mask[b,:curr_len+2] = out_mask_tmp
            out_mask = out_mask.to(device)

            # input to model
            seq_results, bi_class = Enzyme_Model(seq_batch_tokens, repr_layers=[33], return_contacts=False)
            # print(bi_class)
            # print(bi_class.shape)
            # bi_target = torch.empty(bi_class.size(0)).random_(2).long().to(device)
            bi_target = target_class.long()

            # print(torch.softmax(bi_class, dim=-1))
            # print(bi_target)
            # loss_class = F.binary_cross_entropy_with_logits(bi_class, bi_target)
            loss_class = loss_ce_mean(torch.softmax(bi_class, dim=-1), bi_target.to(seq_batch_tokens.dtype))
            # print(loss_class)


            token_logits = seq_results["logits"]
            # print(token_logits.shape) # 1, L_max+2, 33
            token_logits_softmax = torch.softmax(token_logits, dim=-1)
            # print(token_logits.shape, seq_batch_tokens[:,0,:].shape)

            # token_pred = torch.argmax(token_logits_softmax, dim=-1)
            # # print(token_pred.shape)
            # print("".join([alphabet_all_toks[i] for i in seq_batch_tokens[0]]))
            # print("".join([alphabet_all_toks[i] for i in ori_seq_batch_tokens[0]]))
            # print("".join([alphabet_all_toks[i] for i in token_pred[0]]))
            # acc_arr_temp = (ori_seq_batch_tokens==token_pred).int()[:,1:-1]
            # acc = acc_arr_temp.sum()/acc_arr_temp.numel()
            # print(acc)
            # print(seq_mask.shape, out_mask.shape, token_logits.shape, seq_batch_tokens.shape)


            loss_mask = loss_ce((seq_mask.unsqueeze(-1)*out_mask.unsqueeze(-1)*token_logits).reshape(-1,33), (seq_mask*out_mask*seq_batch_tokens).reshape(-1).long())
            loss_mask = loss_mask.mean()
            
            # loss = loss_mask + loss_class
            loss = loss_class

            # except:
            #     print(f"continue_{bt}")
            #     torch.cuda.empty_cache()
            #     continue

            # compute loss
            sum_loss += loss
            avg_loss = sum_loss/(i+1)
            sum_loss_mask += loss_mask
            avg_loss_mask = sum_loss_mask/(i+1)
            sum_loss_class += loss_class
            avg_loss_class = sum_loss_class/(i+1)

            # if i % args.log_freq == 0:
            if i % 50== 0:
                w_lines_base = f"ep={ep} bt={i}/{bts}, avg_loss_class={avg_loss_class:.4f}, avg_loss_mask={avg_loss_mask:.4f}, avg_loss={avg_loss:.4f}"
                print(w_lines_base)
                with open(args.saved_dir+"/log.txt","a") as f0:
                    w_lines = f"{w_lines_base}\n"
                    f0.writelines(w_lines)

            # eval the model
            if i % 100 == 0:
                Enzyme_Model = Enzyme_Model.eval()
                # avg_pl, avg_pl2, avg_pl5 = eval_rosetta_on_fly(esm_model, seq_batch_converter, eval_batch_list, device)
                
                batch_raw = PlasticDB_batch_list
                # print(batch)
                fixed_len = args.fixed_len # 256
                max_len= fixed_len
                crop_true_len = [min(len(seq), max_len) for clas_id, seq in batch_raw]
                target_class = torch.tensor([clas_id for clas_id, seq in batch_raw]).to(device)
                seq_data = [(str(idx),seq[:crop_true_len[idx]]) for idx, (class_id,seq) in enumerate(batch_raw)]

                seq_batch_labels, seq_batch_strs, seq_batch_tokens_tmp = seq_batch_converter(seq_data)
                
                seq_batch_tokens = torch.ones(seq_batch_tokens_tmp.size(0), fixed_len+2) # '<pad>': 1

                bt_max_len = max(crop_true_len)+2
                assert bt_max_len==seq_batch_tokens_tmp.size(1)
                seq_batch_tokens[:,:bt_max_len] = seq_batch_tokens_tmp

                seq_batch_tokens = seq_batch_tokens.to(device=device, dtype=seq_batch_tokens_tmp.dtype)

                with torch.no_grad():
                    acc, pos_acc, neg_acc = Enzyme_Model.infer_class(seq_batch_tokens, repr_layers=[33], return_contacts=False, target_class=target_class)

                Enzyme_Model = Enzyme_Model.train()

                # w_lines_base = f"ep_bt={ep}_{i}, acc={acc:.4f}, P@L={avg_pl:.4f}, P@L2={avg_pl2:.4f}, P@L5={avg_pl5:.4f}, avg_loss={avg_loss:.4f}"
                w_lines_base = f"ep_bt={ep}_{i}, acc={acc:.4f}, pos_acc={pos_acc:.4f}, neg_acc={neg_acc:.4f}, avg_loss={avg_loss:.4f}"
                print(w_lines_base)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            i+=1

        # finish an epoch
        saved_root = f"{args.saved_dir}/checkpoints"
        saved_pt_path = f"{saved_root}/{str(ep).zfill(3)}_{str(i).zfill(5)}_checkpoint.pt"
        torch.save(Enzyme_Model.state_dict(), saved_pt_path)

        # # eval the model
        # Enzyme_Model = Enzyme_Model.eval()
        # # avg_pl, avg_pl2, avg_pl5 = eval_rosetta_on_fly(esm_model.esm_seq_model, seq_batch_converter, eval_batch_list, device, contact_head)
        # avg_pl, avg_pl2, avg_pl5 = eval_rosetta_on_fly(esm_model, seq_batch_converter, eval_batch_list, device)
        # Enzyme_Model = Enzyme_Model.train()

        # with open(args.saved_dir+"/log.txt","a") as f0:
        #     w_lines_base = f"END_ep_bt={ep}_{i}, P@L={avg_pl:.4f}, P@L2={avg_pl2:.4f}, P@L5={avg_pl5:.4f}, avg_loss={avg_loss:.4f}"
        #     w_lines = f"{w_lines_base}\n"
        #     print(w_lines_base)
        #     f0.writelines(w_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir", type=str, default="output/", help="path of saved model")

    parser.add_argument("--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    
    parser.add_argument("--save_freq", type=int, default=1000, help="saving model every n steps")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--optimizer_type", type=str, default="adamw", help="sgd, adam, adamw")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")

    parser.add_argument("--restore_model", type=str, default=None, help="to restore a trained model")
    parser.add_argument("--esm_seq_type", type=str, default="esm2", help="esm1b, esm2")

    parser.add_argument("--train_data_type", type=str, default="huawei_a3m_dataset", help="debug, huawei_a3m_dataset")
    parser.add_argument("--eval_data_type", type=str, default="rosetta_dataset", help="debug, eval_for_train, rosetta_dataset")
    
    parser.add_argument("--mode", type=str, default="train", help="train, eval")
    parser.add_argument("--fixed_len", type=int, default=256, help="fix the length of all sequences in batches")

    args = parser.parse_args()

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    import esm_local as bink_esm
    # print(bink_esm.__file__)

    # Load ESM-Seq model
    if args.esm_seq_type =="esm1b":
        seq_model, seq_alphabet = bink_esm.pretrained.esm1b_t33_650M_UR50S()
    if args.esm_seq_type =="esm2":
        seq_model, seq_alphabet = bink_esm.pretrained.esm2_t33_650M_UR50D()
    print(f"Pretrained seq_model({args.esm_seq_type}) is loaded successfully! ")
    
    # fix the ESM model
    for name, param in seq_model.named_parameters():
        param.requires_grad = False
    
    # print key modelinfo.
    for name, param in seq_model.named_parameters():
        if param.requires_grad:
            print(name)
        else:
            print("xx", name)
    
    Enzyme_Model = EnzymeModel(esm_seq_model=seq_model,
                                embed_dim=1280,
                                fixed_len=args.fixed_len+2)  # why +2? <cls>:the first token and <eos>: the last token

    if args.restore_model:
        restore_model_path = args.restore_model
        Enzyme_Model.load_state_dict(torch.load(restore_model_path), strict=True)
        print(f"Enzyme_Model parameters is loaded successfully (from {restore_model_path})!")

    if args.mode=="train":
        checkpoints_saved_dir = args.saved_dir+"/checkpoints"
        # e.g., "output/checkpoints/"
        if not os.path.exists(checkpoints_saved_dir):
            os.makedirs(checkpoints_saved_dir)

        Enzyme_Model.train()
        Enzyme_Model = Enzyme_Model.to(device)
        train_esm_enzyme(Enzyme_Model, seq_alphabet, device, args)
    
    elif args.mode=="eval":
        Enzyme_Model = Enzyme_Model.eval()
        Enzyme_Model = Enzyme_Model.to(device)

        # for Contact Map evaluation
        # if args.eval_data_type=="debug":
        #     eval_batch_list = load_eval_dataset(split_name="debug")
        # elif args.eval_data_type=="rosetta_dataset":
        #     eval_batch_list = load_eval_dataset(split_name="rosetta_dataset")
        # elif args.eval_data_type=="eval_for_train":
        #     eval_batch_list = load_eval_dataset(split_name="eval_for_train")
        # else:
        #     raise("wrong args.eval_data_type")
        # avg_pl, avg_pl2, avg_pl5 =  eval_rosetta_raw_esm(Enzyme_Model.esm_seq_model, seq_batch_converter, eval_batch_list, device)
        
        seq_batch_converter = seq_alphabet.get_batch_converter()
        # uniref_batch_list = load_train_dataset(split_name="debug")
        PlasticDB_batch_list = load_plastic_dataset(split_name="PlasticDB")

        # batch_raw = PlasticDB_batch_list + uniref_batch_list[:5000] # 182 plastic enzymes and 5k other proteins for evaluation
        batch_raw = PlasticDB_batch_list # 182 plastic enzymes
        # batch_raw = uniref_batch_list[:20000]
        
        fixed_len = args.fixed_len # 256
        max_len= fixed_len
        crop_true_len = [min(len(seq), max_len) for clas_id, seq in batch_raw]
        target_class = torch.tensor([clas_id for clas_id, seq in batch_raw]).to(device)
        seq_data = [(str(idx),seq[:crop_true_len[idx]]) for idx, (class_id,seq) in enumerate(batch_raw)]

        seq_batch_labels, seq_batch_strs, seq_batch_tokens_tmp = seq_batch_converter(seq_data)
        
        seq_batch_tokens = torch.ones(seq_batch_tokens_tmp.size(0), fixed_len+2) # '<pad>': 1

        bt_max_len = max(crop_true_len)+2
        assert bt_max_len==seq_batch_tokens_tmp.size(1)
        seq_batch_tokens[:,:bt_max_len] = seq_batch_tokens_tmp

        seq_batch_tokens = seq_batch_tokens.to(device=device, dtype=seq_batch_tokens_tmp.dtype)

        with torch.no_grad():
            acc, pos_acc, neg_acc = Enzyme_Model.infer_class(seq_batch_tokens, repr_layers=[33], return_contacts=False, target_class=target_class)
        print(acc, pos_acc, neg_acc)

        # pos_acc=0.989010989010989
        # neg_acc=0.9996

    else:
        raise(f"Wrong args.mode (ERR\"{args.mode}\")")