import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la
import math
from scipy.fftpack import fft,fftfreq
from scipy.linalg import toeplitz
import soundfile as sf
import wavio
#Question 1
'''
import numpy as np
import matplotlib.pyplot as plt
theta=np.arange(-10,10,0.1)
L=theta**2
L=np.array(L)
plt.plot(theta,L)
plt.show()
'''
#Question 2
'''
from mpl_toolkits import mplot3d 
import numpy as np 
import matplotlib.pyplot as plt 
fig = plt.figure() 
# syntax for 3-D projection 
ax = plt.gca(projection ='3d') 
# defining all 3 axes 
x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1) 
x,y= np.meshgrid(x,y)
z=x**2+y**2
coor=np.argwhere(z == np.min(z))
print("The minimum value for L(theta)="+str(np.min(z)))
print("theta0="+str(x[coor[0][0]][coor[0][1]])+"\n theta1="+str(y[coor[0][0]][coor[0][1]]))
ax.plot_surface(x,y,z,cmap='Blues') 
plt.show() 
'''

#Question 3
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('d:/Machine learning/Assign1.csv')
X =df.iloc[:,0:1]
Y=df.iloc[:,1:2]
X=np.array(X)
Y=np.array(Y)
theta0=np.arange(0,50,0.5)
theta1=np.arange(-1,1,0.001)
z=np.zeros((len(theta0),len(theta1)))
fig = plt.figure() 
ax = plt.gca(projection ='3d')
for i in range(len(theta0)):
    for j in range(len(theta1)):
        for k in range(len(X)):
            z[i][j]=z[i][j]+((Y[k]-(X[k]*theta1[j])-theta0[i])**2)
        
theta1,theta0= np.meshgrid(theta1,theta0)
coor=np.argwhere(z == np.min(z))

print("The minimum value for L(theta)="+str(np.min(z)))
print("theta0="+str(theta0[coor[0][0]][coor[0][1]])+"\n theta1="+str(theta1[coor[0][0]][coor[0][1]]))

#plt.plot(X,Y,'*',color='black')
#plt.plot(X,theta0[coor[0][0]][coor[0][1]]+theta1[coor[0][0]][coor[0][1]]*X)

#plt.show()


ax.plot_surface(theta0,theta1,z,cmap='Blues')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('L(theta)')

plt.show() 
'''
#Question 4
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('d:/Machine learning/Assign1.csv')
X =df.iloc[0:4,0:1]
Y=df.iloc[0:4,1:2]
X=np.array(X)
Y=np.array(Y)
X2=X
X=np.c_[ np.ones(len(X)),X ]
XT=X.transpose()

temp=np.dot(XT,X)
temp=np.linalg.pinv(temp)
temp2=np.dot(XT,Y)
theta=np.dot(temp,temp2)
print(theta)
'''
'''
#Question 5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('d:/Machine learning/Assign1.csv')
X =df.iloc[:,0:1]
Y=df.iloc[:,1:2]
X=np.array(X)
Y=np.array(Y)
theta=np.zeros((2,1))
theta[0]=45.22
theta[1]=-8.09e-03
X=np.c_[ np.ones(len(X)),X ]
temp=np.matmul(X,theta)
temp=temp-Y
temp=np.matmul(temp.transpose(),temp)
print("L[theta] = "+str(np.sum(temp)))

theta[0]=40
theta[1]=-.05
temp=np.matmul(X,theta)
temp=temp-Y
temp=np.matmul(temp.transpose(),temp)
print("L[theta] = "+str(np.sum(temp)))
'''