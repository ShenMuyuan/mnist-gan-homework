import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 15)
plt.rcParams['image.cmap'] = 'gray'


def show_images(images, cols=4):
    rows = int(np.ceil(len(images) / cols))

    for index, image in enumerate(images):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image.reshape(28, 28))

    plt.show()


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# load the generator model which has been trained for 200 epochs
G = torch.load('Generator_epoch_200.pth')
G.eval()

# sample 20 noises
n_noise = 20
noise = (torch.rand(n_noise, 128) - 0.5) / 0.5
noise = noise.to(device)

# create 20 fake images with the generator
fake_image = G(noise)
imgs_numpy = (fake_image.data.cpu().numpy() + 1.0) / 2.0
show_images(imgs_numpy)
