import torch
import torch.nn as nn

def gradient_penalty(critic,real,fake,device='cpu'):
    batch_size,c,h,w=real.shape
    # torch.tensor.repeat
    # Repeats this tensor along the specified dimensions.
    epsilon=torch.rand((batch_size,1,1,1)).repeat(1,c,h,w).to(device)
    interpolate_images=real*epsilon+fake*(1-epsilon)

    #calculate critic scores
    mixed_scores=critic(interpolate_images)

    # torch.autograd.grad
    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    gradient=torch.autograd.grad(
        inputs=interpolate_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=torch.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm -1)**2)

    return gradient_penalty
