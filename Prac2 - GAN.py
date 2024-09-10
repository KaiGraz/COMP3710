import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime


# With support from https://realpython.com/generative-adversarial-networks/

torch.manual_seed(69)

# LOCATION = "keras_png_slices_data/keras_png_slices_test"
LOCATION = "keras_png_slices_data"

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])
train_data = torchvision.datasets.ImageFolder(LOCATION, data_transform)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Extract the first batch of images
data_iter = iter(train_data_loader)
images, labels = next(data_iter)

class Discriminator(nn.Module):
    size = 256
    out_size = 16384
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.out_size, 4*self.size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4*self.size, 2*self.size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2*self.size, self.size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.out_size)
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    in_size = 100
    size = 256
    out_size = 16384
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.in_size, self.size),
            nn.ReLU(),
            nn.Linear(self.size, 2*self.size),
            nn.ReLU(),
            nn.Linear(2*self.size, 4*self.size),
            nn.ReLU(),
            nn.Linear(4*self.size, 16384),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 128, 128)
        return output


discriminator = Discriminator().to(device)
generator = Generator().to(device)

lr = 0.0001
num_epochs = 1
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

losses = []

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_data_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device)
        latent_space_samples = torch.randn((batch_size, Generator.in_size)).to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, Generator.in_size)).to(device)


        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        losses.append((loss_discriminator.item(), loss_generator.item()))
    
    saving = True     
    # Look at the generated things
    latent_space_samples = torch.randn(batch_size, Generator.in_size).to(device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(128, 128), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    if saving:
        now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        name = "GAN_evidence/NewBrains" + now + ".png"
        plt.savefig(name)
    plt.show()
print("Done")

plt.plot(losses)
plt.show()
if saving:
    now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
    name = "GAN_evidence/NewPlot" + now + ".png"
    plt.savefig(name)
