import torch
import sys

from utils import *


def train(train_loader, model, loss_projection, loss_testing, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    losses_p = AverageMeter()
    losses_t = AverageMeter()
    accs = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss and accuracy
        r = model.encoder(images)
        with torch.no_grad():
            r2 = model.encoder(images)
        z = model.projector(r)
        c = model.classifier(r2)

        loss_p = loss_projection(z, labels)
        loss_t = loss_testing(c, labels)

        loss = 0.8 * loss_p + 0.2 * loss_t

        acc = accuracy(c, labels)

        # update metric
        losses.update(loss.item(), bsz)
        losses_p.update(loss_p.item(), bsz)
        losses_t.update(loss_t.item(), bsz)
        accs.update(acc.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('\tTrain: [{0}][{1}/{2}]\t'
                  'loss ({loss.avg:.3f})\t'
                  'loss ({lossp.avg:.3f})\t'
                  'loss t ({losst.avg:.3f})\t'
                  'acc ({accs.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), loss=losses,
                   lossp=losses_p, losst=losses_t, accs=accs))
            sys.stdout.flush()
            

def validate(test_loader, model, task_id):
    """one epoch training"""
    model.eval()

    accs = AverageMeter()

    for images, labels in test_loader:

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss and accuracy
        c = model.classify(images)

        acc = accuracy(c, labels)

        # update metric
        accs.update(acc.item(), bsz)

    print('\tTest: [{0}]\t'
            'acc ({accs.avg:.3f})\t'.format(
            task_id, accs=accs))
    sys.stdout.flush()