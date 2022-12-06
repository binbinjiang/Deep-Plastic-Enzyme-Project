# Deep Plastic Enzyme Project

## Hello, Lingrong!
This is a private repository related to Deep Plastic Enzyme. Here, I will (only) share interesting and useful codes with you and keep updating this repository. 

Now, this is just the beginning. I believe that your great and meaningful plastic enzyme plan must come true one day, and you can make the world a better place in the future.

Util that that day comes, I hope I can make more small efforts to enlighten you in my way! Whaever you need, I will do my best to support you. It is not just for you but also for all of humanity (/doge, hahahaha). Any questions, just feel free to contact me!

## Features
- A high-accuray binary classification for Plastic Enzymes: Input a protein sequence (usually in a ".fasta" form), then a judgment of Plastic Enzyme will return. 1: Plastic Enzyme; 0: Others.
- Finetuning/Training an ESM-x model is supported. The public ESM project just provides test/inference procedures based on a series of pretrained parameters. Here, I add a simple Finentuning/Training pipeline.
- A Masked language modeling (MLM) method is provided for training a protein language model from the scratch. 
- A MLM loss based on cross-entropy and a masked technique using 'torch.bernoulli' function (which tells you how to create masks for the imported batches) are available.


## Prerequisite
You can use this repository easily. As a prerequisite, you can install the same configurations as ESM repo. Please refer to this [website](https://github.com/facebookresearch/esm/) for more details. E.g., 
```
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```


## Usage
- Train a classification model based on the ESM model. To be simple and lightweight, all the ESM-x model layers are frozen and only the classifier layers can be trained.
```
export CUDA_VISIBLE_DEVICES=0 # specify a GPU number
python train_protein_enzyme.py --batch_size 5 --epochs 10 --saved_dir ./output/enzyme_model/ --esm_seq_type esm2 --train_data_type debug --mode train
```

NOTE: It is easy to converge. In my experiments, I obtained a ~99.0 accuracy for plastic enzyme classification. The pretrained model we may send you via email.

- And a evaluation of plastic enzyme classification as:
```
python train_protein_enzyme.py --esm_seq_type esm2 --train_data_type debug --mode eval --restore_model xxxx.pt
```

- Also, you can train your own protein language model from the scratch (but I don't think you need this):
```
python train_protein_enzyme.py --batch_size 32 --epochs 100 --saved_dir ./output/esm_lm/ --esm_seq_type esm2 --train_data_type uniref50 --mode eval --restore_model xxxx.pt
```

- Evaluate the the pretrained ESM Model. We provide a Contact Map Prediction pipeline as:
```
python train_protein_enzyme.py --batch_size 32 --esm_seq_type esm2 --mode eval --restore_model xxxx.pt
```


## Databases
- A general protein dataset, "Uniref50", including 300M protein sequences in total. It is used for training an ESM-like modle from the scratch

- A Plastic DB, "PlasticDB.fasta", including 182 Plastic Enzyme sequences in total. A core database for deep plastic enzyme research. 


## TODO
1. Collect a complete Enzyme database including oxidoreductases (EC1), transferases (EC2), hydrolases (EC3), lyases (EC4), isomerases (EC5), ligases (EC6) and  translocases (EC7). (Around the corner)

2. Based on the above Enzyme database, a mutil-class classification model is expected to be extended based on the current model.

3. Too many things to do.

## Statement 
THis repository is for Lingrong only! For her means for the entire of the human!