import torch
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda")


def advance_to(M,rays):
    Mr = rays.view(-1,4).matmul(M.transpose(0,1)).view(-1,2,4).transpose(1,2)
    rMr = rays.bmm(Mr)

    D= rMr[:,0,1]*rMr[:,0,1]+rMr[:,1,0]*rMr[:,1,0]+2*rMr[:,0,1]*rMr[:,1,0]-4*rMr[:,0,0]*rMr[:,1,1]

    t0 = -rMr[:,0,0]/(rMr[:,0,1] + rMr[:,1,0])
    t1 = (-rMr[:,0,1]-rMr[:,1,0] + torch.sqrt(D))/(2*rMr[:,1,1])
    t2 = (-rMr[:,0,1]-rMr[:,1,0] - torch.sqrt(D))/(2*rMr[:,1,1])

    t=torch.where(rMr[:,1,1]==0.0, t0, torch.where(t1<0, t2, torch.where(t2<0,t1,torch.min(t1,t2))))

    rays[:,0,:] += t.view(1,-1).transpose(0,1)*rays[:,1,:]
    return rays

def normals_on(M,rays):
    rp = rays[:,0,:]
    normals = rp.matmul(M) + M.matmul(rp.transpose(0,1)).transpose(0,1)
    normals[:,3]=0

    nb = normals.view(-1,1,4)
    norm = torch.sqrt(torch.bmm(nb,nb.transpose(1,2)))
    normals/=norm.view(-1,1)
    return normals


def reflect(M, rays, plot=True):
    normals = normals_on(M,rays)

    nl = normals.view(-1,1,4)
    overlap = torch.bmm(nl,rays[:,1,:].view(-1,4,1)).view(-1,1)
    projection = normals*overlap.view(-1,1)

    reflection = rays[:,1,:]-2*projection
    rays[:,1,:]=reflection
    if plot:
        plot_arrows(rays, normals)
        plot_arrows(rays, projection, "r")
        plot_arrows(rays, reflection, "g")
    return rays

def plot_arrows(rays,v,color="k"):
    
    x = rays[:,0,0].cpu().numpy()
    y = rays[:,0,2].cpu().numpy()
    dx= v[:,0].cpu().numpy()
    dy= v[:,2].cpu().numpy()
    for x,y,dx,dy in zip (x,y,dx,dy):
        plt.arrow(x,y,dx,dy,color=color)

def plot_rays(rays,factor=1.0,color="k"):
    r = rays.cpu().detach().numpy()
    x = r[:,0,0]
    y = r[:,0,2]
    dx= factor * r[:,1,0]
    dy= factor * r[:,1,2]
    for x,y,dx,dy in zip (x,y,dx,dy):
        plt.arrow(x,y,dx,dy,color=color,head_width=0.05,head_length=0.1)

def advance_to_and_reflect(M,rays, plot=False):
    advance_to(M, rays)
    reflect(M, rays, plot=plot)
    return rays


rays = torch.tensor([[[x,y,4,1],[0,0,-0.5,0]] for x in np.linspace(-0.9,0.9,20) for y in np.linspace(-0.1,0.1,3)],dtype=torch.float, device=device)
mirror=torch.tensor(    [
      [ 0.2, 0  , 0  , 0  ],
      [ 0  , 0.2, 0  , 0  ],
      [ 0  , 0  , 0  ,-0.5],
      [ 0  , 0  ,-0.5, 0  ]
    ], requires_grad=True, dtype=torch.float, device=device)
plane=torch.tensor(    [
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  ,-1  ],
      [ 0  , 0  ,-1  , 2  ]
    ], requires_grad=True, dtype=torch.float, device=device)

plot_rays(rays,color="r")
advance_to_and_reflect(mirror,rays)
plot_rays(rays,color="g")
advance_to(plane,rays)
plot_rays(rays,color="b")

plt.xlim(-2,2)
plt.ylim(-1,6)
plt.show()