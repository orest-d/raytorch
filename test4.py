import torch
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cpu")
scalartype = torch.float32

def uMv(u,M,v):
    return (u.matmul(M).view(-1,1,4).bmm(v.view(-1,4,1)))

def advance_to(M, positions, vectors):
    pMp = uMv(positions, M, positions)
    vMp = uMv(vectors, M, positions)
    pMv = uMv(positions, M, vectors)
    vMv = uMv(vectors, M, vectors)

    B = vMp+pMv
    D= B**2-4*pMp*vMv

    t0 = -pMp/B
    t1 = (-B + torch.sqrt(D))/(2*vMv)
    t2 = (-B - torch.sqrt(D))/(2*vMv)

    t=t0                                 #TODO: where did not work with sqrt and autograd
    positions = positions + t.view(1,-1).transpose(0,1)*vectors
    return positions, vectors

def normals_on(M, positions, vectors):
    P = torch.eye(4,4,device=device,dtype=scalartype)
    P[3,3]=0
    normals = positions.matmul(M) + M.matmul(positions.transpose(0,1)).transpose(0,1)
    normals = normals.matmul(P)

    nb = normals.view(-1,1,4)
    norm = torch.sqrt(torch.bmm(nb,nb.transpose(1,2)))
    normals=normals/norm.view(-1,1)
    return normals


def reflect(M, positions, vectors, plot=True):
    normals = normals_on(M, positions, vectors)

    nl = normals.view(-1,1,4)
    overlap = torch.bmm(nl,vectors.view(-1,4,1)).view(-1,1)
    projection = normals*overlap.view(-1,1)

    reflection = vectors-2*projection
    if plot:
        plot_rays(positions, vectors, normals)
        plot_rays(positions, vectors, projection, "r")
        plot_rays(positions, vectors, reflection, "g")
    return positions, reflection

def plot_rays(positions, vectors, factor=1.0,color="k"):
    r = positions.cpu().detach().numpy()
    x = r[:,0]
    y = r[:,2]
    r = vectors.cpu().detach().numpy()
    dx= factor * r[:,0]
    dy= factor * r[:,2]
    for x,y,dx,dy in zip (x,y,dx,dy):
        plt.arrow(x,y,dx,dy,color=color,head_width=0.05,head_length=0.1)

def advance_to_and_reflect(M, positions, vectors, plot=False):
    positions, vectors = advance_to(M, positions, vectors)
    positions, vectors = reflect(M, positions, vectors, plot=plot)
    return positions, vectors


mirror=torch.tensor(    [
      [ 0.01, 0.01  , 0  , 0  ],
      [ 0.01  , 0.05, 0  , 0  ],
      [ 0  , 0  , 0  ,-0.5],
      [ 0  , 0  ,-0.5, 0  ]
    ], requires_grad=True, dtype=scalartype, device=device)
plane=torch.tensor(    [
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  ,-1  ],
      [ 0  , 0  ,-1  , 2  ]
    ], requires_grad=False, dtype=scalartype, device=device)
center = torch.tensor(
      [ 0  , 0  , 1  , 1  ],
    requires_grad=False, dtype=scalartype, device=device)

def rays():
    span = [(x,y) for x in np.linspace(-0.9,0.9,200) for y in np.linspace(-0.9,0.9,200)]
    #span = [(x,y) for x in np.linspace(-0.9,0.9,3) for y in np.linspace(-0.1,0.1,1)]
    positions = torch.tensor([[x,y,4,1] for x,y in span],dtype=scalartype, device=device)
    vectors = torch.tensor([[0,0,-0.5,0] for x,y in span],dtype=scalartype, device=device)
    return positions, vectors

positions, vectors = rays()
plot_rays(positions, vectors, color="r")

start = time.time()
for i in range(100):
    positions, vectors = rays()
    positions, vectors = advance_to_and_reflect(mirror, positions, vectors)
    positions, vectors = advance_to(plane,positions, vectors)

    d = positions-center
    dist = (d*d).sum()
    print(i,dist)
    dist.backward()

    rate = 0.0000015
    g=mirror.grad.cpu().numpy()
    mirror = mirror.detach().cpu()
    m = mirror.numpy()
    g[3,:]=0
    g[:,3]=0
    g[2,:]=0
    g[:,2]=0
    m-=g*rate
    mirror = torch.tensor(m, requires_grad=True, dtype=scalartype, device=device)
end = time.time()

print(mirror)
print(end-start)
#plot_rays(positions, vectors, color="g")
#plt.plot([0],[1],"k+")
#plt.xlim(-2,2)
#plt.ylim(-1,6)
#plt.show()