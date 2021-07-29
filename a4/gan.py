from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)



class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        x = nn.functional.relu(self.fc1(z))
        return torch.sigmoid(self.fc4(x))

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc4 = nn.Linear(400, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return torch.sigmoid(self.fc4(x))

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    mnist_dim = 784
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    criterion = nn.BCELoss(reduction='sum')

    for batch_idx, (x, _) in enumerate(train_loader):
        discriminator.zero_grad()

        # train discriminator on real
        x_real, y_real = x.view(-1, mnist_dim), torch.ones(100, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output = discriminator(x_real)
        D_real_loss = criterion(D_output, y_real)
        D_real_score = D_output

        # train discriminator on facke
        z = Variable(torch.randn(100, 20).to(device))
        x_fake, y_fake = generator(z), Variable(torch.zeros(100, 1).to(device))

        D_output = discriminator(x_fake)
        D_fake_loss = criterion(D_output, y_fake)
        D_fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        discriminator_optimizer.step()
        avg_discriminator_loss += D_loss.data.item()

        #=======================Train the generator=======================#
        generator.zero_grad()

        z = Variable(torch.randn(100, 20).to(device))
        y = Variable(torch.ones(100, 1).to(device))

        G_output = generator(z)
        D_output = discriminator(G_output)
        G_loss = criterion(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        generator_optimizer.step()
        avg_generator_loss += G_loss.data.item()

    avg_generator_loss /= len(train_loader)
    avg_discriminator_loss /= len(train_loader)
    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    mnist_dim = 784
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    criterion = nn.BCELoss(reduction='sum')
    for batch_idx, (x, _) in enumerate(test_loader):

        # train discriminator on real
        x_real, y_real = x.view(-1, mnist_dim), torch.ones(100, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output = discriminator(x_real)
        D_real_loss = criterion(D_output, y_real)

        # train discriminator on facke
        z = Variable(torch.randn(100, 20).to(device))
        x_fake, y_fake = generator(z), Variable(torch.zeros(100, 1).to(device))

        D_output = discriminator(x_fake)
        D_fake_loss = criterion(D_output, y_fake)

        D_loss = D_real_loss + D_fake_loss
        avg_discriminator_loss += D_loss.data.item()

        y = Variable(torch.ones(100, 1).to(device))

        G_loss = criterion(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        avg_generator_loss += G_loss.data.item()

    avg_generator_loss /= len(test_loader)
    avg_discriminator_loss /= len(test_loader)
    return avg_generator_loss, avg_discriminator_loss

epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
