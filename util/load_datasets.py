import  json
import numpy as np
import tqdm
import os

import itertools
import string
from typing import List, Tuple
from Bio import SeqIO

from Bio.PDB.PDBParser import PDBParser
pdb_parser = PDBParser(QUIET=True)
import torch
from biotite.database import rcsb
from biotite.structure.io.pdbx import PDBxFile, get_structure
import biotite.structure as bs
from scipy.spatial.distance import squareform, pdist, cdist
try:
    from contact_metric import evaluate_prediction, contacts_from_pdb, extend
except:
    from .contact_metric import evaluate_prediction, contacts_from_pdb, extend

def load_train_dataset(split_name):
    assert split_name in ["debug", "uniref50"]
    batch_list = []

    # MAX_LEN: max length of each sqeuence
    MAX_LEN= 1024-2

    seq_file_path="/usr/commondata/local_public/forJiangbin/TrainStaticEmb/uniref50.fasta"
    class_id= 0
    with open(seq_file_path) as f:
        raw_lines = f.readlines()
    skip_num = 0
    valid_num = 0
    for line in tqdm.tqdm(raw_lines):
        if valid_num==20000 and split_name=="debug":
            break # for test

        if ">" in line: 
            continue
        else:
            l_tmp = line.strip("\n")
            if len(l_tmp) <= MAX_LEN:
                batch_list.append((class_id, l_tmp))
                valid_num+=1
            else:
                skip_num+=1

    total_num = skip_num + valid_num
    print(f"Load train_dataset successfully ({split_name})\nskip_num={skip_num}/{total_num}, valid_num={valid_num}/{total_num}")
    return batch_list

def load_plastic_dataset(split_name="PlasticDB"):
    batch_list = []

    # MAX_LEN: max length of each sqeuence
    MAX_LEN= 1024-2

    seq_file_path="/root/enzyme_model_bink/PlasticDB.fasta"
    class_id= 1
    with open(seq_file_path) as f:
        raw_lines = f.readlines()
    skip_num = 0
    valid_num = 0
    for line in tqdm.tqdm(raw_lines):
        if ">" in line: 
            continue
        else:
            l_tmp = line.strip("\n")
            if len(l_tmp) <= MAX_LEN:
                batch_list.append((class_id, l_tmp))
                valid_num+=1
            else:
                skip_num+=1

    total_num = skip_num + valid_num
    print(f"Load train_dataset successfully ({split_name})\nskip_num={skip_num}/{total_num}, valid_num={valid_num}/{total_num}")
    return batch_list


def load_eval_dataset(split_name):
    assert split_name in ["debug", "rosetta_dataset", "eval_for_train"]

    eval_batch_list = []
    # root_path = "/usr/commondata/local_public/rep/train/fasta"
    root_path = "/usr/commondata/local_public/forJiangbin/fasta"
    skip_num = 0
    valid_num = 0
    pdbname_seq_arr = []
    for fasta_dir in os.listdir(root_path):
        fasta_dir_path = os.path.join(root_path, fasta_dir)
        # print(fasta_dir_path)
        fasta_path = fasta_dir_path+"/"+fasta_dir+".fasta"
        # print(fasta_path)
        with open(fasta_path,"r",encoding="utf-8") as f:
            lines = f.readlines()
        # name = lines[0][1:].strip("\n")
        # seq = lines[1].strip("\n")[0:1024-2]  # maximum  sequence length of 1024
        seq = lines[1].strip("\n")
        if len(seq)>1024-2:
            skip_num+=1
            continue
        else:
            valid_num+=1
        pdbname_seq_arr.append((fasta_dir, seq))
    total_num = skip_num + valid_num
    print(f"Load eval_dataset successfully ({split_name})\nskip_num={skip_num}/{total_num}, valid_num={valid_num}/{total_num}")

    # load ground-truth contacts
    if split_name=="debug":
        pdbname_seq_arr = pdbname_seq_arr[:10]
    elif split_name=="eval_for_train":
        pdbname_seq_arr = pdbname_seq_arr[:320]

    for pdbname, seq in tqdm.tqdm(pdbname_seq_arr):
        # true_pdbfile = os.path.join("/usr/commondata/local_public/rep/train/pdb", pdbname+".pdb")
        true_pdbfile = os.path.join("/usr/commondata/local_public/ProteinRelated/15051Dataset/pdb", pdbname+".pdb")
        structure = pdb_parser.get_structure(pdbname, true_pdbfile)
        
        Cb_coords = np.zeros((len(seq),3))
        for num, chain in enumerate(structure[0]):
            assert num==0
            for index, residue in enumerate(chain):
                try:
                    atom_Cb = residue['CB']
                    Cb_coords[index] = atom_Cb.get_coord()
                except:
                    atom_Ca = residue['CA']
                    atom_C = residue['C']
                    atom_N = residue['N']
                    # print(atom_C.get_coord(), atom_N.get_coord(), atom_Ca.get_coord())
                    atom_Cb = extend(atom_C.get_coord(), atom_N.get_coord(), atom_Ca.get_coord(), 1.522, 1.927, -2.143)
                    Cb_coords[index] = atom_Ca.get_coord()
        
        t1 = np.expand_dims(Cb_coords, axis=-2)
        t2 = np.expand_dims(Cb_coords, axis=-3)
        dist = np.sqrt(np.sum((t1-t2)**2, axis=-1))

        distance_threshold = 8
        contacts = dist < distance_threshold
        contacts = contacts.astype(np.int64)
        contacts[np.isnan(dist)] = -1

        eval_batch_list.append((contacts, seq))

    return eval_batch_list

if __name__=="__main__":
    print("0")
    # load_train_dataset(split_name="debug_dataset", N_SEQ=64)

    eval_batch_list = load_eval_dataset("debug")
    for eval_bt in eval_batch_list:
        contact_ref, seq = eval_bt
        print(len(seq), contact_ref.shape)
        predict_contact = torch.tensor(contact_ref).bool()
        metrics = evaluate_prediction(predict_contact, contact_ref)
        print(metrics["long_P@L"],metrics["long_P@L2"],metrics["long_P@L5"])