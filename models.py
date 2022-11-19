import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

class Encoder(nn.Module):

    def __init__(self, input_dim=784, r_dim=400) -> None:
        super(Encoder,self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, r_dim),
            nn.ReLU(),
            nn.Linear(r_dim, r_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Projector(nn.Module):

    def __init__(self, r_dim=400, z_dim=10) -> None:
        super(Projector,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(r_dim, z_dim),
        )

    def forward(self, r):
        z_raw = self.layers(r)
        z = F.normalize(z_raw, dim=1)
        return z


class Oja(nn.Module):

    def __init__(self, z_dim=10) -> None:
        super(Oja,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 1),
        )

    def forward(self, z):
        return self.layers(z)


class Classifier(nn.Module):

    def __init__(self, r_dim=400, num_classes=10) -> None:
        super(Classifier,self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(r_dim, num_classes)
        )

    def forward(self, r):
        return self.layers(r)


class Filter(nn.Module):

    def __init__(self) -> None:
        super(Filter,self).__init__()
        self.dims = []

    def append(self, d):
        self.dims.append(d)

    def forward(self, z):
        for _, d in enumerate(self.dims):
            # this mat mul is doing the dot product for each vector in the mini batch
            z = z - d*torch.matmul(z, torch.atleast_2d(d).T)
            z = F.normalize(z, dim=1)
        
        return z


class SupervisedContrastiveLearner(nn.Module):

    def __init__(self) -> None:
        super(SupervisedContrastiveLearner,self).__init__()

        self.encoder = Encoder()
        self.projector = Projector()
        self.filter = Filter()
        self.classifier = Classifier()
        self.classifier_loss = CrossEntropyLoss()
        self.classifier_optimiser = SGD(self.classifier.parameters(), lr=1e-3)

    def forward(self, x):
        return self.filter(self.projector(self.encoder(x)))

    def train_classifier(self, r, labels):
        self.classifier_optimiser.zero_grad()
        y = self.classifier(r)
        loss = self.classifier_loss(y, (labels==labels[0]).long())
        loss.backward()
        self.classifier_optimiser.step()

    def classify(self, x):
        return self.classifier(self.encoder(x))