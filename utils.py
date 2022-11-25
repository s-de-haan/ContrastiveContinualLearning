import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class Options:
    print_freq: int        # how many iterations before another print
    batch_size: int        # batch size
    num_workers: int       # cores to be able to expand on training
    temp: float            # temperature parameter of the SupCon loss function
    learning_rate: float   # learning rate
    momentum: float        # SGD momentum parameter
    weight_decay: float    # SGD weight decay
    epochs: int            # number of training epochs


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output,1)
        batch_size = target.size(0)
        correct = torch.sum(pred.eq(target))
        
        return 100 * correct / batch_size