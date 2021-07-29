from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = not torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.reLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc_encode = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_size)
        self.fc_var = nn.Linear(400, latent_size)

        self.fc_decode1 = nn.Linear(latent_size, 400)
        self.fc_decode2 = nn.Linear(400, 784)

    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        h1 = self.reLU(self.fc_encode(x))
        return self.fc_mu(h1), self.fc_var(h1)

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        std = torch.exp(0.5*log_variances)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(means) # return z sample

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        h1 = self.reLU(self.fc_decode1(z))
        return self.sigmoid(self.fc_decode2(h1)) 

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_x, x.view(-1, 784), reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
    loss = reconstruction_loss + KL_divergence
    return loss, reconstruction_loss



def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    model.train()
    avg_train_loss = 0
    avg_train_reconstruction_loss = 0
    for batch_index, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            tmp_loss, tmp_reconstruction_loss = vae_loss_function(recon_batch, data, mu, log_var)
            tmp_loss.backward()
            avg_train_loss += tmp_loss.item()
            avg_train_reconstruction_loss += tmp_reconstruction_loss.item()
            optimizer.step()

            if batch_index % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f} \t Reconstruction Loss: {:.6f}'.format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                    100. * batch_index / len(train_loader), tmp_loss.item(), tmp_reconstruction_loss.item()))

    avg_train_loss /= len(train_loader)
    avg_train_reconstruction_loss /= len(train_loader)
    return avg_train_loss, avg_train_reconstruction_loss


def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    model.eval()
    avg_test_loss = 0
    avg_test_reconstruction_loss = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        recon_batch, mu, logvar = model(data)
        tmp_loss, tmp_reconstruction_loss = vae_loss_function(recon_batch, data, mu, logvar)
        avg_test_loss += tmp_loss.item()
        avg_test_reconstruction_loss += tmp_reconstruction_loss.item()
    
    print('\nEpoch {}: Test set: Average loss: {:.6f}, Average reconstruction loss: {:.6f} \n'.format(epoch, 
        100. * avg_test_loss / len(test_loader.dataset), 100. * avg_test_reconstruction_loss / len(test_loader.dataset)))
    
    avg_test_loss /= len(train_loader)
    avg_test_reconstruction_loss /= len(train_loader)
    return avg_test_loss, avg_test_reconstruction_loss

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

fig_train_loss = plt.figure()
plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
fig_train_loss.savefig('Train_loss.png')

fig_test_loss = plt.figure()
plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
fig_test_loss.savefig('Test_loss.png')
