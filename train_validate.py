import torch
import datetime
import time
import pandas as pd
import copy
from matplotlib import pyplot as plt
import transformer_torch as trto

ngpu = trto.ngpu
use_cuda = torch.cuda.is_available()  
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")

transformer = trto.get_model()
optimizer = trto.get_optimizer()
lr_scheduler = trto.get_lr_scheduler()

EPOCHS = 80
print_trainstep_every = 50
metric_name = 'acc'
df_history = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])

def train_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]
    # enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)
    enc_padding_mask, combined_mask, dec_padding_mask = trto.create_mask(inp, targ_inp)
    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)
    # print('device:', inp.device, targ_inp)
    model.train()
    optimizer.zero_grad()  
    # forward
    prediction, _ = transformer(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = trto.mask_loss_func(targ_real, prediction)
    metric = trto.mask_accuracy_func(targ_real, prediction)
    # backward
    loss.backward()  
    optimizer.step() 
    return loss.item(), metric.item()

# batch_src, batch_targ = next(iter(trto.train_dataloader)) # [64,10], [64,10]
# print(train_step(transformer, batch_src, batch_targ))
"""
x += pos_encoding  # [b, inp_seq_len, d_model]
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
"""

def validate_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]
    # enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)
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

def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)

def train_model(model, epochs, train_dataloader, val_dataloader, print_every):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_acc = 0.
    for epoch in range(1, epochs + 1):

        # lr_scheduler.step() # 

        loss_sum = 0.
        metric_sum = 0.

        for step, (inp, targ) in enumerate(train_dataloader, start=1):
            # inp [64, 10] , targ [64, 10]
            loss, metric = train_step(model, inp, targ)

            loss_sum += loss
            metric_sum += metric


            if step % print_every == 0:
                print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, {metric_name}: {metric_sum / step:.3f}')

            lr_scheduler.step()  # 

        # test(model, train_dataloader)
        val_loss_sum = 0.
        val_metric_sum = 0.
        for val_step, (inp, targ) in enumerate(val_dataloader, start=1):
            loss, metric = validate_step(model, inp, targ)
            val_loss_sum += loss
            val_metric_sum += metric


        # record = (epoch, loss_sum/step, metric_sum/step)
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # print('*'*8, 'EPOCH = {} loss: {:.3f}, {}: {:.3f}'.format(
        #        record[0], record[1], metric_name, record[2]))
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
            record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # current_acc_avg = metric_sum / step
        current_acc_avg = val_metric_sum / val_step #
        if current_acc_avg > best_acc:  #
            best_acc = current_acc_avg
            # checkpoint = save_dir + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())  ##################
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, "project2_model.pt")

    print('finishing training...')
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return df_history

def plot_metric(df_history, metric):
    plt.figure()
    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig('./imgs/' + metric + 'pair.png')  # save img
    plt.show()

trto.draw_pos_encoding(trto.pos_encoding)
df_history = train_model(transformer, EPOCHS, trto.train_dataloader, trto.val_dataloader, print_trainstep_every)
print(df_history)
plot_metric(df_history, 'loss')
plot_metric(df_history, metric_name)
