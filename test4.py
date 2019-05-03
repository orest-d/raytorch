import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")

#def uMv(u,M,v):
#    print (u)
#    print (u.matmul(M))
#    print (u.matmul(M).matmul(v.transpose(0,1)))
#def uMv(u,M,v):
#    print (u)
#    print (u.matmul(M))
#    print ("uM",u.matmul(M).view(-1,1,3))
#    print ("v",v.view(-1,3,1))
#    print (u.matmul(M).view(-1,1,3).bmm(v.view(-1,3,1)))

def uMv(u,M,v):
#    print ("u",u)
#    print ("uM",u.matmul(M).view(-1,1,4))
    return (u.matmul(M).view(-1,1,4).bmm(v.view(-1,4,1)))

#uMv(
#    torch.tensor([[1,0,0],[0,10,0],[1,0,0],[0,10,0],[1,0,0],[0,10,0]]),
#    torch.tensor([[1,2,3],[4,5,6],[7,8,9]]),
#    torch.tensor([[11,12,13],[14,15,16],[1,0,0],[0,10,0],[1,0,0],[0,10,0]])
#    )   

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

#    t=torch.where(vMv==0.0, t0, torch.where(t1<0, t2, torch.where(t2<0,t1,torch.min(t1,t2))))
    t=t0
#    print ("positions",positions.size())
#    print ("vectors",vectors.size())
#    print ("pMp",pMp.size())
#    print ("vMp",vMp.size())
#    print ("pMv",pMv.size())
#    print ("vMv",vMv.size())
#    print ("t",t.size())
#    print (t.view(1,-1).transpose(0,1)*vectors)
    positions = positions + t.view(1,-1).transpose(0,1)*vectors
    return positions, vectors

def normals_on(M, positions, vectors):
    P = torch.eye(4,4,device=device)
    P[3,3]=0
    normals = positions.matmul(M) + M.matmul(positions.transpose(0,1)).transpose(0,1)
    print("normals",normals)
    normals = normals.matmul(P)
    print("normals",normals)

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

span = [(x,y) for x in np.linspace(-0.9,0.9,20) for y in np.linspace(-0.9,0.9,20)]
#span = [(x,y) for x in np.linspace(-0.9,0.9,3) for y in np.linspace(-0.1,0.1,1)]
positions = torch.tensor([[x,y,4,1] for x,y in span],dtype=torch.float, device=device)
vectors = torch.tensor([[0,0,-0.5,0] for x,y in span],dtype=torch.float, device=device)

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
    ], requires_grad=False, dtype=torch.float, device=device)
center = torch.tensor(
      [ 0  , 0  , 1  , 1  ],
    requires_grad=False, dtype=torch.float, device=device)


plot_rays(positions, vectors, color="r")
positions, vectors = advance_to_and_reflect(mirror, positions, vectors)
plot_rays(positions, vectors, color="g")
positions, vectors = advance_to(plane,positions, vectors)

d = positions-center
dist = (d*d).sum()
print(d)
print(dist)
dist.backward()
print (mirror.grad)

plot_rays(positions,vectors,color="b")
plt.plot([0],[1],"k+")
plt.xlim(-2,2)
plt.ylim(-1,6)
plt.show()