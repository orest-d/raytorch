import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    t=torch.where(vMv==0.0, t0, torch.where(t1<0, t2, torch.where(t2<0,t1,torch.min(t1,t2))))

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


plane=torch.tensor(    [
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  ,-1  ],
      [ 0  , 0  ,-1  ,-4  ]
    ], requires_grad=False, dtype=scalartype, device=device)

W=320
H=200
def rays():
    span = [(x,y) for x in np.linspace(-3.2,+3.2,W) for y in np.linspace(-2.0,2.0,H)]
    #span = [(x,y) for x in np.linspace(-0.9,0.9,3) for y in np.linspace(-0.1,0.1,1)]
    positions = torch.tensor([[0,0,0,1] for x,y in span],dtype=scalartype, device=device)
    vectors = torch.tensor([[x,1,y,0] for x,y in span],dtype=scalartype, device=device)
    return positions, vectors

start = time.time()
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

positions, vectors = rays()
p=vectors.cpu().numpy()
print(p)
ax.plot(xs=0.5*p[:,0],ys=0.5*p[:,1],zs=0.5*p[:,2],marker="o",linestyle="")
ax.plot(xs=p[:,0],ys=p[:,1],zs=p[:,2],marker="o",linestyle="")

positions, vectors = advance_to_and_reflect(plane, positions, vectors)
p=positions.cpu().numpy()
ax.plot(xs=p[:,0],ys=p[:,1],zs=p[:,2],marker="o",linestyle="")
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-5,5)

pic = np.zeros((W,H)).reshape(W*H)
index = (p[:,1]<0)&(p[:,2]<1)
#p[:,0].trunc()+p[:,1].trunc()
pic[index]=(p[index,0].round()+p[index,1].round())%2
print (index)

#(positions[:,0]+positions[:,1])%1

end = time.time()

print(end-start)
#plot_rays(positions, vectors, color="g")
#plt.plot([0],[1],"k+")
#plt.xlim(-2,2)
#plt.ylim(-1,6)
plt.show()
plt.imshow(pic.reshape((W,H)).T)
plt.show()
