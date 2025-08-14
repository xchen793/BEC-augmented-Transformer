import torch
import random
from matplotlib import pyplot as plt
import transformer_torch as trto

ngpu = 1
use_cuda = torch.cuda.is_available()  
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
tokenizer = lambda x: x.split() 

checkpoint = "project2_model.pt"
print('checkpoint:', checkpoint)
ckpt = torch.load(checkpoint) 
transformer_sd = ckpt['net']
optimizer_sd = ckpt['opt'] 
lr_scheduler_sd = ckpt['lr_scheduler']

reload_model = trto.get_model()

print('Loading model ...')
reload_model.load_state_dict(transformer_sd)
print('Model loaded ...')

def tokenizer_encode(tokenize, sentence, vocab):
    # print(type(vocab)) # torchtext.vocab.Vocab
    # print(len(vocab))
    sentence = trto.normalizeString(sentence)
    # print(type(sentence)) # str
    sentence = tokenize(sentence)  # list
    sentence = ['<start>'] + sentence + ['<end>']
    sentence_ids = [vocab.stoi[token] for token in sentence]
    # print(sentence_ids, type(sentence_ids[0])) # int
    return sentence_ids

def tokenzier_decode(sentence_ids, vocab):
    sentence = [vocab.itos[id] for id in sentence_ids if id<len(vocab)]
    # print(sentence)
    return " ".join(sentence)

# s = 'je pars en vacances pour quelques jours .'
# print(tokenizer_encode(tokenizer, s, SRC_TEXT.vocab))

# s_ids = [3, 5, 251, 17, 365, 35, 492, 390, 4, 2]
# print(tokenzier_decode(s_ids, SRC_TEXT.vocab))
# print(tokenzier_decode(s_ids, TARG_TEXT.vocab))

def evaluate(model, inp_sentence):
    model.eval() 

    inp_sentence_ids = tokenizer_encode(tokenizer, inp_sentence, trto.SRC_TEXT.vocab)  
    # print(tokenzier_decode(inp_sentence_ids, SRC_TEXT.vocab))
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len=10]
    # print(encoder_
    # input.shape)

    decoder_input = [trto.TARG_TEXT.vocab.stoi['<start>']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]
    # print(decoder_input.shape)

    with torch.no_grad():
        for i in range(trto.MAX_LENGTH + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = trto.create_mask(encoder_input.cpu(), decoder_input.cpu()) ################
            # [b,1,1,inp_seq_len], [b,1,targ_seq_len,inp_seq_len], [b,1,1,inp_seq_len]

            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            # forward
            predictions, attention_weights = model(encoder_input,
                                                   decoder_input,
                                                   enc_padding_mask,
                                                   combined_mask,
                                                   dec_padding_mask)
            # [b=1, targ_seq_len, target_vocab_size]
            # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
            #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


            prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
            prediction_id = torch.argmax(prediction, dim=-1)  # => [b=1, 1]
            # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
            if prediction_id.squeeze().item() == trto.TARG_TEXT.vocab.stoi['<end>']:
                return decoder_input.squeeze(dim=0), attention_weights

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]


    return decoder_input.squeeze(dim=0), attention_weights
    # [targ_seq_len],
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}



# ==== BLEU metric (dependency-free) ==========================================
import math
from collections import Counter

SPECIAL_TOKENS = {"<start>", "<end>", "<pad>", "<unk>"}

def _clean_and_tokenize(text: str):
    """Normalize like your pipeline, then whitespace-tokenize."""
    if text is None:
        return []
    text = trto.normalizeString(text)
    return [t for t in tokenizer(text) if t not in SPECIAL_TOKENS]

def _ids_to_tokens_remove_special(ids, vocab):
    toks = [vocab.itos[i] for i in ids if i < len(vocab)]
    toks = [t for t in toks if t not in SPECIAL_TOKENS]
    return toks

def _ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

def _modified_precision(candidate, references, n):
    """
    Clipped n-gram precision for a single candidate against one or more references.
    Returns (clipped_count, total_count).
    """
    cand_counts = Counter(_ngrams(candidate, n))
    if not cand_counts:
        return 0, 0

    # Max ref counts across refs (standard BLEU definition)
    max_ref_counts = Counter()
    for ref in references:
        ref_counts = Counter(_ngrams(ref, n))
        for ng, cnt in ref_counts.items():
            if cnt > max_ref_counts[ng]:
                max_ref_counts[ng] = cnt

    clipped = 0
    for ng, cnt in cand_counts.items():
        clipped += min(cnt, max_ref_counts.get(ng, 0))
    total = sum(cand_counts.values())
    return clipped, total

def _brevity_penalty(c, r):
    if c == 0:
        return 0.0
    if c > r:
        return 1.0
    return math.exp(1.0 - float(r) / float(c))

def sentence_bleu(candidate_tokens, reference_tokens_list, max_n=4, smoothing_k=1.0):
    """
    Candidate: list[str]; references: list[list[str]]
    Returns BLEU in [0,1].
    """
    precisions = []
    for n in range(1, max_n+1):
        clipped, total = _modified_precision(candidate_tokens, reference_tokens_list, n)
        # simple add-k smoothing (Chen & Cherry style variants); k=1 works well for short sents
        p_n = (clipped + smoothing_k) / (total + smoothing_k) if total > 0 else 0.0
        precisions.append(p_n)

    # Brevity penalty uses closest reference length
    c = len(candidate_tokens)
    ref_lens = [len(ref) for ref in reference_tokens_list]
    r = min(ref_lens, key=lambda rl: (abs(rl - c), rl)) if ref_lens else 0

    bp = _brevity_penalty(c, r)
    if any(p == 0 for p in precisions):
        # if any precision is zero after smoothing=0, score would be zero;
        # with add-k, they won't be zero unless total==0. Keep numeric safety:
        pass
    score = bp * math.exp(sum((1.0/max_n) * math.log(p) for p in precisions if p > 0))
    return score

def corpus_bleu(candidates, list_of_references, max_n=4, smoothing_k=1.0):
    """
    Corpus BLEU with aggregated clipped counts (recommended).
    candidates: List[List[str]]
    list_of_references: List[List[List[str]]]  (each item is a list of refs; here we use one ref)
    """
    total_clipped = [0]*max_n
    total_counts  = [0]*max_n
    cand_len_sum = 0
    ref_len_sum  = 0

    for cand, refs in zip(candidates, list_of_references):
        cand_len_sum += len(cand)
        # closest reference length per sentence
        if refs:
            rlens = [len(r) for r in refs]
            ref_len_sum += min(rlens, key=lambda rl: (abs(rl - len(cand)), rl))
        else:
            rlens = [0]
            ref_len_sum += 0

        for n in range(1, max_n+1):
            clipped, total = _modified_precision(cand, refs, n)
            total_clipped[n-1] += clipped
            total_counts[n-1]  += total

    precisions = []
    for n in range(1, max_n+1):
        clipped = total_clipped[n-1]
        total   = total_counts[n-1]
        # add-k smoothing
        p_n = (clipped + smoothing_k) / (total + smoothing_k) if total > 0 else 0.0
        precisions.append(p_n)

    bp = _brevity_penalty(cand_len_sum, ref_len_sum)
    score = bp * math.exp(sum((1.0/max_n) * math.log(p) for p in precisions if p > 0))
    return score

# ---- Helpers to evaluate your current samples --------------------------------
def evaluate_sentence_bleu(src_sentence: str, ref_sentence: str):
    pred_ids, _ = evaluate(reload_model, src_sentence)
    cand_tokens = _ids_to_tokens_remove_special(pred_ids.tolist(), trto.TARG_TEXT.vocab)
    ref_tokens  = _clean_and_tokenize(ref_sentence)
    return sentence_bleu(cand_tokens, [ref_tokens])

def evaluate_corpus_bleu(sentence_pairs):
    cands = []
    refs  = []
    for src, ref in sentence_pairs:
        pred_ids, _ = evaluate(reload_model, src)
        cand_tokens = _ids_to_tokens_remove_special(pred_ids.tolist(), trto.TARG_TEXT.vocab)
        ref_tokens  = _clean_and_tokenize(ref)
        cands.append(cand_tokens)
        refs.append([ref_tokens])  # list of references per example
    return corpus_bleu(cands, refs)

# ==== END BLEU ================================================================


s = 'je pars en vacances pour quelques jours.'
evaluate(reload_model,s)

s = 'je pars en vacances pour quelques jours.'
s_targ = '	i m taking a couple of days off .'
pred_result, attention_weights = evaluate(reload_model, s)
pred_sentence = tokenzier_decode(pred_result, trto.TARG_TEXT.vocab)
print('real target:', s_targ)
print('pred_sentence:', pred_sentence)



sentence_pairs = [
    ['je pars en vacances pour quelques jours .', 'i m taking a couple of days off .'],
    ['je ne me panique pas .', 'i m not panicking .'],
    ['je recherche un assistant .', 'i am looking for an assistant .'],
    ['je suis loin de chez moi .', 'i m a long way from home .'],
    ['vous etes en retard .', 'you re very late .'],
    ['j ai soif .', 'i am thirsty .'],
    ['je suis fou de vous .', 'i m crazy about you .'],
    ['vous etes vilain .', 'you are naughty .'],
    ['il est vieux et laid .', 'he s old and ugly .'],
    ['je suis terrifiee .', 'i m terrified .'],
]

def batch_translate(sentence_pairs):
    for pair in sentence_pairs:
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, _ = evaluate(reload_model, pair[0])
        pred_sentence = tokenzier_decode(pred_result, trto.TARG_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')

# batch_translate(sentence_pairs)

def evaluateRandomly(n=10):
    for i in range(n):
        pair = random.choice(trto.pairs)
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, attentions = evaluate(reload_model, pair[0])
        pred_sentence = tokenzier_decode(pred_result, trto.TARG_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')

# evaluateRandomly(2)

def plot_attention_weights(attention, sentence, pred_sentence, layer):
    sentence = sentence.split()
    pred_sentence = pred_sentence.split()

    fig = plt.figure(figsize=(16, 8))

    # block2 attention[layer] => [b=1, num_heads, targ_seq_len, inp_seq_len]
    attention = torch.squeeze(attention[layer], dim=0) # => [num_heads, targ_seq_len, inp_seq_len]

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)  #

        cax = ax.matshow(attention[head].cpu(), cmap='viridis')  #

        fontdict = {'fontsize': 10}


        ax.set_xticks(range(len(sentence)+2))  #
        ax.set_yticks(range(len(pred_sentence)))

        ax.set_ylim(len(pred_sentence) - 1.5, -0.5)  

        ax.set_xticklabels(['<start>']+sentence+['<end>'], fontdict=fontdict, rotation=90)  
        ax.set_yticklabels(pred_sentence, fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()

def translate(sentence_pair, plot=None):
    print('input:', sentence_pair[0])
    print('target:', sentence_pair[1])
    pred_result, attention_weights = evaluate(reload_model, sentence_pair[0])
    print('attention_weights:', attention_weights.keys())
    pred_sentence = tokenzier_decode(pred_result, trto.TARG_TEXT.vocab)
    print('pred:', pred_sentence)
    print('')

    if plot:
        plot_attention_weights(attention_weights, sentence_pair[0], pred_sentence, plot)

batch_translate(sentence_pairs)
evaluateRandomly(2)
translate(sentence_pairs[0], plot='decoder_layer4_block2')

# Compute corpus BLEU on your small eval set
bleu_corpus = evaluate_corpus_bleu(sentence_pairs)
print(f"[Corpus BLEU-4] {bleu_corpus*100:.2f}")

# Also print per-sentence BLEU for a quick look
for src, ref in sentence_pairs:
    s_bleu = evaluate_sentence_bleu(src, ref)
    print(f"src: {src}\nref: {ref}\nBLEU-4: {s_bleu*100:.2f}\n")
