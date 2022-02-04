import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Critic,Generator,initialize_weights

from gradient_penalty import gradient_penalty


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#lr=5e-5
lr=1e-4
batch_size=64
image_size=64
channels_img=1
z_dim=100
n_epochs=5
features_disc=64
features_gen=64
critic_iterations=5
# weight_clip=0.01
lambda_gp=10

transforms=transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)])
    ]
)

dataset=datasets.MNIST(root='PyTorch_basics\data',
                        train=True,
                        transform=transforms)

loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
gen=Generator(z_dim,channels_img,features_gen).to(device)
critic=Critic(channels_img,features_disc).to(device)
initialize_weights(gen)
initialize_weights(critic)

#optimizer_gen=optim.RMSprop(gen.parameters())
#optimizer_critic=optim.RMSprop(critic.parameters())

optimizer_gen=optim.Adam(gen.parameters(),lr=lr,betas=(0.0,0.9))
optimizer_critic=optim.Adam(critic.parameters(),lr=lr,betas=(0.0,0.9))


fixed_noise=torch.randn(32,z_dim,1,1).to(device)
writer_real=SummaryWriter('WGAN/logs/real')
writer_fake=SummaryWriter('WGAN/logs/fake')
step=0

gen.train()
critic.train()

for epoch in range(n_epochs):
    for batch_index, (real,_) in enumerate(loader):
        real=real.to(device)

        for _ in range(critic_iterations):
            noise=torch.randn((batch_size,z_dim,1,1)).to(device)
            fake=gen(noise)

            critic_real=critic(real).reshape(-1)
            critic_fake=critic(fake).reshape(-1)
            
            gp=gradient_penalty(critic,real,fake,device)
            # the discriminator wants to maximize the distance between the distribution
            # but in pytorch optimizers are built for minimizing functions
            loss_critic=(
                -(torch.mean(critic_real)-torch.mean(critic_fake))+lambda_gp*gp
            )

            optimizer_critic.zero_grad()  # disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()

            #for p in critic.parameters():
            #    p.data.clamp_(-weight_clip,weight_clip)

        #train generator
        output=critic(fake).reshape(-1)
        loss_gen=-torch.mean(output)
        optimizer_gen.zero_grad()  #gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_index % 100 == 0:
            print(
                f"Epoch [{epoch}/{n_epochs}] Batch {batch_index}/{len(loader)} \n"
                f"Loss D:{loss_critic:.4f}, Loss G:{loss_gen:.4f}"
            )

            with torch.no_grad():
                fake=gen(fixed_noise).reshape(-1,1,64,64)
                data=real.reshape(-1,1,64,64)

                img_grid_fake=torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real=torchvision.utils.make_grid(data,normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images",img_grid_fake,global_step=step
                )
                writer_real.add_image(
                    'Mnist Real Image',img_grid_real,global_step=step
                )
                step+=1
