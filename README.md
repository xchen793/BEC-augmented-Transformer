# dl_final_project
channel - combined transformer on machine translation

## Abstract
Abstract: In order to improve Transformer performance on large corpus in machine translation, a new architecture design called channel-combined transformer was explored based on Transformer by inserting Binary Erasure Channels and Binary Symmetric Channels between the Encoder and Decoder. Moreover, additive white gaussian noise(AWGN) was used to simulate the noisy perturbation on textual input embeddings. In the experiment, erasure/crossover probability p was fine-tuned in corresponding experiments to compare machine translation performance between the channel-combined transformer and original transformer. Preliminary results revealed that the channel-combined transformer significantly shorten the training time with trivial loss in validation accuracy in both noisy and noise-free conditions. In addition, channel-combined transformer better preserves the semantic meaning than the standard transformer. Future work could be conducted on how to improve Transformer robustness against adversarial attack by leveraging information-theoretic approaches to process the inner-structural information flow, such as rate-distortion theory.

## Methodology


## Preliminary Results: 
1. Computational time dropped by 8% with only 0.4% loss of validation accuracy. 
2. The effect of improving Transformer robustness is still unclear.
2. The channel-combined transformer outperforms original transformer in terms of semantic meaning preservation.
