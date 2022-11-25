import torch
import torch.backends.cudnn as cudnn

from models import *
from losses import *
from utils import *
from train_validate import *
from continual import split_MNIST

print("Parameters")

# Paramters
EPOCHS = 3
TASKS = {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = Options()
opt.print_freq = 100
opt.batch_size = 32         
opt.num_workers = 2
opt.temp = 0.07             
opt.learning_rate = 0.01 
opt.momentum = 0.9
opt.weight_decay = 1e-4
opt.epochs = 2

print("Model")

# Model & Optimizer & Losses
model = Model()    

optimizer = torch.optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

loss_testing = torch.nn.CrossEntropyLoss()
loss_projection = SupConLoss()

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.cuda()
    loss_testing = loss_testing.cuda()
    loss_projection = loss_projection.cuda()
    cudnn.benchmark = True

print("Train")

# Train
for task_id, digits in TASKS.items():
    train_loader, test_loader = split_MNIST(digits, opt)

    # Training
    for epoch in range(1, opt.epochs+1):    
        train(train_loader, model, loss_projection, loss_testing, optimizer, epoch, opt)
        validate(test_loader, model, task_id)

    model.filter.add_dimension(model, train_loader)

    # Validation
    for task_id, digits in TASKS.items():
        _, test_loader = split_MNIST(digits, opt)
        validate(test_loader, model, task_id)