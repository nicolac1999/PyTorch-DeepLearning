from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from zmq import device


class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),# one output : Fake or Real
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)

# the MNIST images will be normalized between -1 and +1 so we have to add
# Tanh as last layer to set the output values between -1 and +1 as the input values
class Generator(nn.Module):
    def __init__(self,z_dim,img_dim): 
        # z_dim : latent noise 
        super().__init__()
        self.gen=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim), #img_dim: 784 (28x28x1)
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.gen(x)

# Hyperparameters
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr=0.0003
z_dim=64 #128,256
image_dim=28*28*1
batch_size=32
num_epochs=50

disc=Discriminator(image_dim).to(device)
gen=Generator(z_dim,image_dim).to(device)

fixed_noise=torch.randn((batch_size,z_dim)).to(device)

transforms=transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]
)

dataset=datasets.MNIST(root='PyTorch_basics\data',
                        transform=torchvision.transforms.ToTensor(),
                        download=False)

# print(len(dataset))   # 60000

loader=DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True)

optimizer_disc=optim.Adam(disc.parameters(),lr=lr)
optimizer_gen=optim.Adam(gen.parameters(),lr=lr)

criterion=nn.BCELoss()

writer_fake=SummaryWriter('GAN/runs/GAN_MINST/fake')
writer_real=SummaryWriter('GAN/runs/GAN_MINST/real')
step=0

for epoch in range(num_epochs):
    for batch_index,(real,_) in enumerate(loader):
        #we don't need the label of the image
        #flatten the image
        real=real.view(-1,784).to(device)
        batch_size=real.shape[0]


        ### Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        ###                      min - log(D(real)) - log(1-D(G(z)))
        noise=torch.randn((batch_size,z_dim)).to(device)
        fake=gen(noise)

        # log(D(real))
        disc_real=disc(real).view(-1)
        lossD_real= criterion(disc_real,torch.ones_like(disc_real))

        # log(1-D(G(z)))
        disc_fake=disc(fake).view(-1)
        lossD_fake= criterion(disc_fake,torch.zeros_like(disc_fake))

        lossD=(lossD_real+lossD_fake)/2
        optimizer_disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()

        ### Train Generator min log(1-D(G(z))) = max log(D(G(z)))
        ### we can reuse fake=gen(noise) but we have to pass fake.detach() or 
        ### lossD.backward(retain_graph=True)
        ### because when we call loss.backward() the part of graph that calculates
        ### the loss will be freed by default to save memory, so we don't have access to fake

        output=disc(fake).view(-1)
        lossG=criterion(output,torch.ones_like(output))
        
        optimizer_gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

        if batch_index == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \n"
                f"Loss D:{lossD:.4f}, Loss G:{lossG:.4f}"
            )

            with torch.no_grad():
                fake=gen(fixed_noise).reshape(-1,1,28,28)
                data=real.reshape(-1,1,28,28)

                img_grid_fake=torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real=torchvision.utils.make_grid(data,normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images",img_grid_fake,global_step=step
                )
                writer_real.add_image(
                    'Mnist Real Image',img_grid_real,global_step=step
                )
                step+=1
                










  