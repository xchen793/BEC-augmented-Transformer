# BEC-augmented transformer on machine translation

## Abstract
In order to improve Transformer performance on large corpus in machine translation, a new architecture design called channel-combined transformer was explored based on Transformer by inserting Binary Erasure Channels and Binary Symmetric Channels between the Encoder and Decoder. Moreover, additive white gaussian noise(AWGN) was used to simulate the noisy perturbation on textual input embeddings. In the experiment, erasure/crossover probability p was fine-tuned in corresponding experiments to compare machine translation performance between the channel-combined transformer and original transformer. Preliminary results revealed that the channel-combined transformer significantly shorten the training time with trivial loss in validation accuracy in both noisy and noise-free conditions. In addition, channel-combined transformer better preserves the semantic meaning than the standard transformer. Future work could be conducted on how to improve Transformer robustness against adversarial attack by leveraging information-theoretic approaches to process the inner-structural information flow, such as rate-distortion theory.

## Methodology
Compared to the standard transformer, we have two modifications: 1. add channel modeling between the encoder and decoder to decrease computation time; 2. add additive white gaussian noise to test the model performance in noisy condition.

## Experimental Results
1. Computational time dropped by 8% with only 0.4% loss of validation accuracy. 
2. The effect of improving Transformer robustness is still unclear.
3. The channel-combined transformer outperforms original transformer in terms of semantic meaning preservation.


## Environment Setup

We recommend using **conda** for environment management.  
You can create the exact environment from the provided `environment.yml` file:

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate [virtual-env]
channel-combined-transformer-finalproject/
│
├── bec/bsc.py             # Implementation script of BEC and BSC 
├── awgn.py                # Implementation script of AWGN
├── transformer_torch      # Transformer architecture 
├── train_validate.py      # Training script for the transformer
├── evaluate.py            # Evaluation script (computes Accuracy, BLEU, etc.)
├── results_plot.py        # Plots validation accuracy and BLEU trends
├── models/                # Transformer architecture and model components
├── data/                  # Dataset and preprocessing scripts
├── imgs/                  # Images of accuracies, losses and the model architecture 
├── results.csv            # Validation accuracies and BLEU scores across BEC p
├── environment.yml        # Conda environment configuration
└── README.md              # Project documentation (this file)

## Paper: Understanding Transformer Encoder–Decoder Representations through Bernoulli Dropout.
In this paper, we study Transformer overparameterization through the lens of angular similarity in high-dimensional encoder–decoder embeddings. We apply Bernoulli dropout between the encoder and the decoder, varying the keep probability p to identify a sparsity-dependent threshold above which the Top-1 prediction is preserved. Theoretically, we prove that, if the effective sparsity embeddings is sufficiently large, and thus decoder performance, remain stable under moderate coordinate dropout. Empirically, we implement the Bernoulli dropout by constructing a new Transformer model augmented with Binary Erasure Channel (BEC) and test its performance on an English–French translation task. Experimental results visualize the trends for validation accuracies and BLEU scores, both decline sharply at some threshold.

