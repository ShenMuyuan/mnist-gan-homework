import torch.nn as nn


class gan_discriminator(nn.Module):
    def __init__(self):
        super(gan_discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # the prediction result is limited between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)


class gan_generator(nn.Module):
    def __init__(self):
        super(gan_generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
