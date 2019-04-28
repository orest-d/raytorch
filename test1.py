import torch
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cpu")
a=torch.tensor([[1,0.01],[0,2]],dtype=torch.float, device=device)

rays = torch.tensor([[[x,0,0,1],[0,1,0,0]] for x in np.linspace(-0.9,0.9,200)],dtype=torch.float, device=device)
print (rays)

scene = torch.tensor([
    [
      [ 1  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  ,-0.5],
      [ 0  , 0  , 0  , 0  ],
      [ 0  ,-0.5, 0  , 0  ]
    ],
    [
      [ 1  , 0  , 0  , 0  ],
      [ 0  , 1  , 0  , 0  ],
      [ 0  , 0  , 0  , 0  ],
      [ 0  , 0  , 0  ,-1  ]
    ]
],dtype=torch.float, device=device)
M=scene[0]
M=scene[1]

print (scene.size())
print (rays.size())
print ("M",M)
#print ("M2",M2)
#rr=M.dot(rays)
rb = rays.view(-1,4)
print (rb)
print("M rays")
Mr = M.matmul(rb.transpose(0,1)).transpose(0,1).view(-1,2,4).transpose(1,2)
print("Mr",Mr.size())
print (Mr)
r=rays
print("r",r.size())
print (r)


rMr = r.bmm(Mr)
print(rMr)
D= rMr[:,0,1]*rMr[:,0,1]+rMr[:,1,0]*rMr[:,1,0]+2*rMr[:,0,1]*rMr[:,1,0]-4*rMr[:,0,0]*rMr[:,1,1]
print(rMr[:,0,0])
print(rMr[:,0,1])
print(rMr[:,1,1])
print(D)
t0 = -rMr[:,0,0]/(rMr[:,0,1] + rMr[:,1,0])
t1 = (-rMr[:,0,1]-rMr[:,1,0] + torch.sqrt(D))/(2*rMr[:,1,1])
t2 = (-rMr[:,0,1]-rMr[:,1,0] - torch.sqrt(D))/(2*rMr[:,1,1])


t=torch.where(rMr[:,1,1]==0.0, t0, torch.where(t1<0, t2, torch.where(t2<0,t1,torch.min(t1,t2))))
print(t)
print(rays.size())
rv = rays[:,1,:]
rp = rays[:,0,:]
#plt.plot(rays[:,0,0].numpy(), rays[:,0,1].numpy(), "bo")
rays[:,0,:] += t.view(1,-1).transpose(0,1)*rv
plt.plot(rays[:,0,0].numpy(), rays[:,0,1].numpy(), "go")

print("rv",rv.size(),rv)
print(t.view(1,-1).size(), t.view(1,-1))
print(t.view(1,-1).transpose(0,1)*rv)

rp = rays[:,0,:]
print ("rp",rp.size(),"\n",rp)
print ("M",M.size(),"\n",M)

normals = rp.matmul(M) + M.matmul(rp.transpose(0,1)).transpose(0,1)
normals[:,3]=0
print ("normals",normals.size(),"\n",normals)
nb = normals.view(-1,1,4)
print ("nb",nb.size(),"\n",nb)
nbT = nb.transpose(1,2)
print ("nbT",nbT.size(),"\n",nbT)

norm = torch.sqrt(torch.bmm(nb,nbT))
print ("norm",norm.size(),"\n",norm)
normals/=norm.view(-1,1)
print ("normals",normals.size(),"\n",normals)
x = rays[:,0,0].numpy()
y = rays[:,0,1].numpy()
dx= normals[:,0].numpy()
dy= normals[:,1].numpy()

for x,y,dx,dy in zip (x,y,dx,dy):
    plt.arrow(x,y,dx,dy)


nl = normals.view(-1,1,4)
print ("nl",nl.size(),"\n",nl)
rvr = rays[:,1,:].view(-1,4,1)
print("rvr",rvr.size(),rvr)

overlap = torch.bmm(nl,rvr).view(-1,1)
print ("overlap",overlap.size(),"\n",overlap)
projection = normals*overlap.view(-1,1)
print ("projection",projection.size(),"\n",projection)

x = rays[:,0,0].numpy()
y = rays[:,0,1].numpy()
dx= projection[:,0].numpy()
dy= projection[:,1].numpy()

for x,y,dx,dy in zip (x,y,dx,dy):
    plt.arrow(x,y,dx,dy,color="r")

reflection = rays[:,1,:]-2*projection
print ("reflection",reflection.size(),"\n",reflection)
rays[:,1,:]=reflection

x = rays[:,0,0].numpy()
y = rays[:,0,1].numpy()
dx= reflection[:,0].numpy()
dy= reflection[:,1].numpy()
for x,y,dx,dy in zip (x,y,dx,dy):
    plt.arrow(x,y,dx,dy,color="g")

#print("M2 rays")
#print (M2.matmul(rb.transpose(0,1)))

x=np.linspace(-0.9,0.9,100)
plt.plot(x,x*x)
plt.plot(x,np.sqrt(1-x*x))


plt.xlim(-2,2)
plt.ylim(-1,3)
plt.show()