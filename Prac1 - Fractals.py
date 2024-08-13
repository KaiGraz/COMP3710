import torch
import numpy as np
import matplotlib.pyplot as plt

# GPU vs CPU Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def part_1():
    # Create 2 grids
    len, width, step = 5, 5, 0.1
    X, Y = np.mgrid[-len:width:step, -len:width:step]

    # Put grids into tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    # Transfer to device
    x = x.to(device)
    y = y.to(device)

    # Compute
    # z = torch.exp(-(x**2+y**2)/2) # Gaussian
    # z = torch.sin(x) # 2D Sine
    # z = torch.cos(x) # 2D Cosine
    z = torch.exp(-(x**2+y**2)/2) * torch.sin(x) # Modulation
    

    # Plot
    plt.imshow(z.cpu().numpy())
    plt.tight_layout()
    plt.show()
    
def part_2(fractal: int = 0, c = 0.285):
    """ Fractal values follow:
    0 - Mandelbrot set,
    1 - Julia set
    
    Where c is added to the Julia set
    """
    # Create 2 grids (but complex)
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    # Y, X = np.mgrid[-0.3:0.3:0.0005, -2:-1:0.0005]
    # Y, X = np.mgrid[-0.1:0.1:0.0002, -1.5:-1.25:0.0002]
    
    # Load everything into tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y) # This makes our grid actually complex
    zs = z.clone()
    ns = torch.zeros_like(z)
    
    # Transfer to device
    # x = x.to(device)
    # y = y.to(device)
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)
    
    # Create the Fractal set over 200 iterations
    for i in range(200):
        # Get new values of z: z^2 + x
        if fractal == 0:
            zs_ = zs*zs + z
        elif fractal == 1:
            zs_ = zs*zs + c
            
        
        # Check for divergence
        if fractal == 0:
            not_diverged = torch.abs(zs_) < 4.0 
        elif fractal == 1:
            not_diverged = torch.abs(zs_) < 2.0
        
        # Update variables to compute
        ns += not_diverged
        zs = zs_
    
    fig = plt.figure(figsize=(8,5))
    
    def processFractal(a):
        """Display an array of iteration counts as a
        colorful picture of a fractal."""
        a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
        img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
        img[a==a.max()] = 0
        a = img
        a = np.uint8(np.clip(a, 0, 255))
        return a
    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


