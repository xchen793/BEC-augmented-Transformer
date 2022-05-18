import torch
import transformer_torch as trto
from transformer_torch import *

ngpu = trto.ngpu
use_cuda = torch.cuda.is_available()  
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")

checkpoint = "0.1_0.1_0.8.pt"
print('checkpoint:', checkpoint)
# ckpt = torch.load(checkpoint, map_location=device) 
ckpt = torch.load(checkpoint) 
# print('ckpt', ckpt)
transformer_sd = ckpt['net']
optimizer_sd = ckpt['opt'] 
lr_scheduler_sd = ckpt['lr_scheduler']

reload_model = trto.get_model()



print('Loading model ...')
reload_model.load_state_dict(transformer_sd)
print('Model loaded ...')

def validate_step(model, inp, targ):
   targ_inp = targ[:, :-1]
   targ_real = targ[:, 1:]
   enc_padding_mask, combined_mask, dec_padding_mask = trto.create_mask(inp, targ_inp)
   inp = inp.to(device)
   targ_inp = targ_inp.to(device)
   targ_real = targ_real.to(device)
   enc_padding_mask = enc_padding_mask.to(device)
   combined_mask = combined_mask.to(device)
   dec_padding_mask = dec_padding_mask.to(device)
   model.eval()  

   with torch.no_grad():
      # forward
      prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
      val_loss = trto.mask_loss_func(targ_real, prediction)
      val_metric = trto.mask_accuracy_func(targ_real, prediction)
   return val_loss.item(), val_metric.item()

def test(model, dataloader):
    # model.eval()

    test_loss_sum = 0.
    test_metric_sum = 0.
    for test_step, (inp, targ) in enumerate(dataloader, start=1):
        # inp [64, 10] , targ [64, 10]
        loss, metric = validate_step(model, inp, targ)
        # print('*'*8, loss, metric)

        test_loss_sum += loss
        test_metric_sum += metric

    print('*' * 8,
          'Test: loss: {:.3f}, {}: {:.3f}'.format(test_loss_sum / test_step, 'test_acc', test_metric_sum / test_step))


#############################################

def tokenizer_encode(tokenize, sentence, vocab):
    # print(type(vocab)) # torchtext.vocab.Vocab
    # print(len(vocab))
    sentence = normalizeString(sentence)
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

def evaluate(model, inp_sentence):
    model.eval()  # 设置eval mode

    inp_sentence_ids = tokenizer_encode(tokenizer, inp_sentence, SRC_TEXT.vocab)  # 转化为索引
    # print(tokenzier_decode(inp_sentence_ids, SRC_TEXT.vocab))
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len=10]
    # print(encoder_input.shape)

    decoder_input = [TARG_TEXT.vocab.stoi['<start>']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]
    # print(decoder_input.shape)

    with torch.no_grad():
        for i in range(MAX_LENGTH + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input.cpu(), decoder_input.cpu()) ################
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

            # 看最后一个词并计算它的 argmax
            prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
            prediction_id = torch.argmax(prediction, dim=-1)  # => [b=1, 1]
            # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
            if prediction_id.squeeze().item() == TARG_TEXT.vocab.stoi['<end>']:
                return decoder_input.squeeze(dim=0), attention_weights

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]
            # decoder_input在逐渐变长

    return decoder_input.squeeze(dim=0), attention_weights
    # [targ_seq_len],
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}



# s = 'je pars en vacances pour quelques jours .'
# evaluate(s)

sentence_pairs = [
    
    ['Vous allez le briser si vous ne faites pas attention .', 'You ll break it if you re not careful .'],
    ['Je n aurais jamais pensé qu il y ait là un tel endroit tranquille dans cette ville bruyante .', 'I never dreamed of there being such a quiet place in this noisy city .'],
    ['Votre voix me semble très familière .', 'Your voice sounds very familiar to me .'],
    ['Presque tout le monde est déjà rentré à la maison .', 'Almost everybody has already gone home .'],


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
        pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')

batch_translate(sentence_pairs)


print('*' * 8, 'final test...')
test(reload_model, trto.val_dataloader)