from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
import torch.nn as nn
from collections import Counter
import pickle
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def get_label_proportion(labels,num_class,packet_size):
    if labels.shape[0]% packet_size== 0:
        batch_size = labels.shape[0]//packet_size
        #print(batch_size)
    else:
        batch_size = labels.shape[0] // packet_size + 1

    distribution = torch.zeros(batch_size,num_class)
    #print(distribution.shape)
    #print(distribution.shape)
    for idx,label in enumerate(labels):
        distribution[idx//packet_size,label]+=1
    if labels.shape[0]% packet_size== 0:
        distribution = distribution / packet_size
    else:
        for i in range(batch_size-1):
            distribution[i] = distribution[i]/packet_size
        distribution[-1] = distribution[-1]/(labels.shape[0]-packet_size*(labels.shape[0]//packet_size))
    #print(distribution)
    #print(torch.sum(distribution,dim=1))
    return distribution

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--packet_size', type=int, default=128,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='100_512_1000.pth',
                        help='path to pre-trained model')

    parser.add_argument('--sample_size', type=int, default=20,
                        help='choose how many samples to optimize')
    parser.add_argument('--thre', type=float, default=0.03,
                        help='choose how many samples to optimize')

    parser.add_argument('--thre2', type=float, default=0.03,
                        help='choose how many samples to optimize')

    parser.add_argument('--model_save_path', type=str, default="model.pkl")
    parser.add_argument('--acc_save_path', type=str, default="acc.pkl")

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '../data/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

class new_classifier(nn.Module):
    def __init__(self,num_class):
        super(new_classifier, self).__init__()
        self.relu = nn.ReLU()
        self.W1 = nn.Linear(2048,1024)
        self.W2 = nn.Linear(1024,num_class)
    def forward(self,feature,adj):
        #f[b,p,2048] adj[b,p,p]
        f = torch.matmul(adj,feature)
        f = self.relu(self.W1(f))
        f = self.W2(f)

        return f

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    #classifier = new_classifier(num_class=opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    entropy = HLoss()
    model.eval()
    classifier.train()
    criterion = nn.KLDivLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    entropy_losses = AverageMeter()
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if labels.shape[0]!=1024:
            break
        #print(labels.shape)
        distribution = get_label_proportion(labels, num_class=opt.n_cls, packet_size=opt.packet_size)

        # obtain class bigger than thre, process batch of bags
        threshold_label = []
        for r in distribution:
            thre_list = []
            for lal,c in enumerate(r):
                if c>opt.thre: #0.03:
                    thre_list.append(lal)
            #if not thre_list:
                #_,top3 = torch.topk(r,3)
                #thre_list.extend(top3.tolist())
            threshold_label.append(thre_list)
        #threshold_label tells which labels are greater than a threshold

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        distribution = distribution.cuda()
        bsz = labels.shape[0]
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)#[1024,2048]
            reshape_features = torch.reshape(features,(int(1024 / opt.packet_size), opt.packet_size, -1))
        output = classifier(features.detach())#[1024,10]
        outputs = F.softmax(output, dim=-1)

        entropy_loss = entropy(output)/(output.shape[0]*output.shape[1])
        #print(entropy_loss)
        _, c_idx = torch.topk(outputs,1,dim=1)
        c_idx = c_idx.unsqueeze(1)#[1,1024]
        reshape_c_idx = torch.reshape(c_idx,(int(1024 / opt.packet_size), opt.packet_size))#[B,bag]
        outputs = torch.reshape(outputs, (int(1024 / opt.packet_size), opt.packet_size, -1))#[b,p,100]

        outputs = torch.mean(outputs, dim=1)


        statistic_label = torch.zeros(int(1024 / opt.packet_size),opt.n_cls)
        for r, rr in enumerate(reshape_c_idx):
            for cc in rr:
                statistic_label[r][cc]+=1
        #value,top = torch.topk(statistic_label,5)

        
        batch_features = []
        for r, rr in enumerate(statistic_label):
            bag_features = []
            for c, c_fre in enumerate(rr):
                if c_fre >= opt.packet_size*opt.thre2:
                    for ttt, lal in enumerate(reshape_c_idx[r]):
                        if lal == c:
                            bag_features.append(reshape_features[r][ttt])
           
            batch_features.append(bag_features)

        outputs = torch.log(outputs)
        loss = criterion(outputs, distribution)
        # update metric
        losses.update(loss.item(), bsz)
        entropy_losses.update(entropy_loss.item(),bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        #print(loss)
        #print(entropy_loss)
        # SGD
        total_loss = loss+0.01*entropy_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'ent_loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, ent_loss=entropy_losses, top1=top1))
            sys.stdout.flush()
        # ------------------------------------------

        nll_loss = nn.NLLLoss()
        for r in range(int(1024/opt.packet_size)):
            if not threshold_label[r] or not batch_features[r]:
                continue
            for lal in threshold_label[r]:
                #print(threshold_label)
                curt_bag_feature = batch_features[r]
                f = torch.stack(curt_bag_feature).cuda()
                tar = torch.tensor([lal for kkk in range(f.shape[0])],dtype=torch.long).cuda()
                #f = cons_features[r] #[20,2048]
                output = classifier(f.detach())  # [1024,10]
                outputs = F.log_softmax(output,dim=-1)
                nll = 0.0005*nll_loss(outputs,tar)
                optimizer.zero_grad()
                #print(nll)
                nll.backward()
                optimizer.step()
                #print(outputs.shape)

        # ------------------------------------------

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            if labels.shape[0] != 1024:
                break
            bsz = labels.shape[0]

            #add
            features = model.encoder(images)  # [1024,2048]
            #print(features.shape)
            #feat = F.normalize(features, dim=1)
            #reshape_features = torch.reshape(feat, (int(1024 / opt.packet_size), opt.packet_size, -1))
            #similarity_1 = torch.matmul(reshape_features, torch.transpose(reshape_features, 1, 2))
            #tempA = 1 / torch.sum(similarity_1, 2)
            #A = torch.zeros(int(1024 / opt.packet_size), opt.packet_size, opt.packet_size).cuda()
            #A[:, range(A.shape[1]), range(A.shape[1])] = tempA[:]
            #A = torch.sqrt(A)
            #A = torch.matmul(torch.matmul(A, similarity_1), A)
            #print(features.shape)
            #reshape_features = torch.reshape(features, (int(1024 / opt.packet_size), opt.packet_size, -1))

            # forward
            #output = classifier(model.encoder(images))
            #print(reshape_features.shape)
            #print(A.shape)
            output = classifier(features)
            #output = output.reshape(1024,-1)#add
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    acc_list = []
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        acc_list.append(val_acc.cpu().numpy())
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), opt.model_save_path)

    with open(opt.acc_save_path,'wb') as f:
        pickle.dump(acc_list,f)
    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
