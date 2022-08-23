import os
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import MELDRobertaCometDataset
from model import MaskedNLLLoss
#from submodels import submodel, submodelsresidual, submodelsresiduals, selfattentiontext
#from attention import  selfattentiontext, submodels, convmodels
from submodels import Simple_LSTM
from sklearn.metrics import f1_score, accuracy_score, classification_report


def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_MELD_loaders(batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaCometDataset('train', classify)
    validset = MELDRobertaCometDataset('valid', classify)
    testset = MELDRobertaCometDataset('test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, audio_feature, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        #print('audio_feature',audio_feature.shape)

        #out, log_prob = model(audio_feature)
        #out, log_prob = model(r1, r2) #r1,r2
        out, log_prob = model(r1, r2, r3, r4, audio_feature) #r1,r2,r3,r4

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
#            if args.tensorboard:
#                for param in model.named_parameters():
#                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()


    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=False, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=23, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=2, help='normalization strategy')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_gru = True
    if args.classify == 'emotion':
        n_classes  = 7
    elif args.classify == 'sentiment':
        n_classes  = 3
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_s = 768
    D_g = 150 #D_g = 150
    D_p = 150 #D_p = 150
    D_r = 150 #D_r = 150
    D_i = 150 #D_i = 150
    D_h = 150 #D_h = 100
    D_a = 100
    D_audio = 768

    D_e = D_p + D_r + D_i

    global seed
    seed = args.seed
    #seed_everything(seed)
    
    #hyparameters
    input_size = 4096
    hidden_size = 256
    hidden_size_fc = 256
    num_layers = 2
    num_classes =7

    model = Simple_LSTM(input_size, hidden_size, hidden_size_fc, num_layers, num_classes).cuda()
 
    print(model)


    if cuda:
        model.cuda()

    if args.classify == 'emotion':
        if args.class_weight:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721])
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
            
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    if args.classify == 'emotion':
        lf = open('logs/cosmic_meld_emotion_logs.txt', 'a')
    elif args.classify == 'sentiment':
        lf = open('logs/cosmic_meld_sentiment_logs.txt', 'a')

    train_loader, valid_loader, test_loader = get_MELD_loaders(batch_size=batch_size, 
                                                               classify=args.classify,
                                                               num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None
    best_score = 0

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)

        if valid_fscore[0] > best_score:
            best_loss, best_label, best_pred, best_mask = test_loss, test_label, test_pred, test_mask
            best_score = valid_fscores[0]

        
        if args.tensorboard:
            writer.add_scalar('train accuracy: ', train_acc, e)
            writer.add_scalar('train loss:', train_loss, e)
            writer.add_scalar('valid accuracy: ', valid_acc, e)
            writer.add_scalar('valid loss:', valid_loss, e)
            writer.add_scalar('test accuracy: ', test_acc, e)
            writer.add_scalar('test loss:', test_loss, e)
        
        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print (x)
        lf.write(x + '\n')

    if args.tensorboard:
        writer.close()
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]    
    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))

    print("Class wise scores:")
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))

    if args.classify == 'emotion':
        rf = open('results/cosmic_meld_emotion_results.txt', 'a')
    elif args.classify == 'sentiment':
        rf = open('results/cosmic_meld_sentiment_results.txt', 'a')
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()


