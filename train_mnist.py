import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from gan_model import gan_discriminator, gan_generator
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
plt.rcParams['figure.figsize'] = (10, 12)
plt.rcParams['image.cmap'] = 'gray'

# Based on https://papers.nips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
# The number of steps to apply to the discriminator, k, is 1


def show_images(images, cols=4):
    rows = int(np.ceil(len(images) / cols))

    for index, image in enumerate(images):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image.reshape(28, 28))

    plt.show()


def d_loss_function(inputs, targets):
    # use BCE loss to get the difference between predictions and labels.
    # minimize this loss to maximize the probability of assigning correct
    # label to both training examples and samples from generator
    return nn.BCELoss()(inputs, targets)


def g_loss_function(inputs):
    # set targets to 1 to make BCE only calculate negative log prediction.
    # we want all fake images considered real by the discriminator
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    # minimize the negative log to better 'cheat' the discriminator
    return nn.BCELoss()(inputs, targets)


# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Model
G = gan_generator().to(device)
D = gan_discriminator().to(device)
print(G)
print(D)

# Settings
epochs = 200
lr = 0.0002
batch_size = 64
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Display some random images from MNIST
random_indices = torch.randint(low=0, high=len(train_set), size=(24,))
random_images = []
for index in random_indices:
    random_images += [train_set[index][0]]
show_images(random_images)

# Training epochs. In every epoch the algorithm scans the entire training set
for epoch in range(epochs):
    epoch += 1
    fake_inputs = None

    #
    # Training iterations. In every iteration the algorithm processes one mini batch
    #
    # First we sample mini batch of batch_size real images. Original label (number of image) is
    # not used here; the new label is that the images coming from MNIST are all real (1), and
    # the images coming from generator are all fake (0).
    #
    for times, (images, _) in enumerate(train_loader):
        times += 1

        real_inputs = images.to(device)
        real_inputs = real_inputs.view(-1, 784)
        # forward pass in discriminator to get prediction
        real_outputs = D(real_inputs)
        # all labels of real images are 1
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        # sample mini batch of batch_size noise
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        # forward pass in generator to get fake images
        fake_inputs = G(noise)
        # forward pass in discriminator to get prediction
        fake_outputs = D(fake_inputs)
        # all labels of noise generated images are 0
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        # Concatenate all outputs (prediction) and all targets (label)
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        # Backpropagation for discriminator (first equation in algorithm)
        d_optimizer.zero_grad()
        d_loss = d_loss_function(outputs, targets)
        d_loss.backward()
        d_optimizer.step()

        # again, sample mini batch of batch_size noise
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        # forward pass in generator to get fake images
        fake_inputs = G(noise)
        # forward pass in discriminator to get prediction
        fake_outputs = D(fake_inputs)

        # Backpropagation for generator (second equation in algorithm)
        g_optimizer.zero_grad()
        g_loss = g_loss_function(fake_outputs)
        g_loss.backward()
        g_optimizer.step()

        if times % 100 == 0 or times == len(train_loader):
            print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader),
                                                                        d_loss.item(), g_loss.item()))

    imgs_fake = (fake_inputs.data.cpu().numpy() + 1.0) / 2.0
    show_images(imgs_fake[:16])

    if epoch % 50 == 0:
        torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

print('Training Finished.')
print('Cost Time: {}s'.format(time.time() - start_time))
