import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import os


# With support from https://realpython.com/generative-adversarial-networks/

torch.manual_seed(69)

LOCATION = "/home/groups/comp3710/OASIS" if os.path.exists("/home/groups/comp3710/OASIS") else "keras_png_slices_data"
OUT_LOCATION = "/home/Student/s4696386/GAN_evidence" if os.path.exists("/home/Student/s4696386/GAN_evidence") else "GAN_evidence"

lr = 0.0001
num_epochs = 10
batch_size = 64


device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])
train_data = torchvision.datasets.ImageFolder(LOCATION, data_transform)

# Filter out all samples from unwanted classes
classes_to_remove = [train_data.class_to_idx[i] for i in train_data.classes if "seg" in i]
filtered_samples = [sample for sample in train_data.samples if sample[1] != classes_to_remove]
train_data.samples = filtered_samples

# Loads data into dataloader, and applies data transformations
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input shape: [batch_size, 1, 128, 128]
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 64, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=8),  # Output: [batch_size, 1, 1, 1]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, 1] for binary classification

class Generator(nn.Module):
    in_size = 100
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input latent vector size: [batch_size, 100, 1, 1]
            nn.ConvTranspose2d(100, 512, kernel_size=8, stride=1, padding=0),  # Output: [batch_size, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 1, 128, 128]
            nn.Tanh()  # Output in range [-1, 1] (assuming normalized images)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 100, 1, 1)  # Reshape input latent vector to [batch_size, 100, 1, 1]
        return self.model(x)


discriminator = Discriminator().to(device)
generator = Generator().to(device)

loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

losses = []

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_data_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device)

        # Implements label smoothing by using 0.9 and 0.1
        real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9
        latent_space_samples = torch.randn((batch_size, Generator.in_size)).to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((generated_samples.size(0), 1)).to(device) + 0.1
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

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
    # Save the generated images
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
        name = OUT_LOCATION + "/Brains" + now + ".png"
        plt.savefig(name)
        plt.clf()
        

plt.figure()  # Start a new figure to avoid overlapping with subplots
discriminator_losses, generator_losses = zip(*losses)  # Unzip the losses
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.plot(generator_losses, label="Generator Loss")
plt.ylim(-2, 2) # Prevents large losses from obscuring details
plt.legend()
if saving:
    now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
    name = OUT_LOCATION + "/NewPlot" + now + ".png"
    plt.savefig(name)

print("Done")