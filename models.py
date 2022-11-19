import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_dim=748, r_dim=400) -> None:
        super(Encoder,self).__init__()

        self.layers = nn.Seqential(
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

        self.layers = nn.Seqential(
            nn.Linear(r_dim, z_dim),
        )

    def forward(self, r):
        z_raw = self.layers(r)
        z = F.normalize(z_raw, dim=1)
        return z


class Oja(nn.Module):

    def __init__(self, z_dim=10) -> None:
        super(Oja,self).__init__()

        self.layers = nn.Seqential(
            nn.Linear(z_dim, 1),
        )

    def forward(self, z):
        return self.layers(z)


class Classifier(nn.Module):

    def __init__(self, r_dim=400, num_classes=2) -> None:
        super(Classifier,self).__init__()

        self.layers = nn.Seqential(
            nn.Linear(r_dim, num_classes),
        )

    def forward(self, r):
        return self.layers(r)


class Filter(nn.Module):

    def __init__(self) -> None:
        self.dims = []

    def append(self, d):
        self.dims.append(d)

    def forward(self, z):
        for _, d in enumerate(self.dims):
            z = z - d*torch.dot(z, d)
            z = F.normalize(z, dim=1)
        
        return z