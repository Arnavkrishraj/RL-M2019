
# coding: utf-8

# ### Import Statements

# In[27]:


import numpy as np
from scipy import optimize


# ## Q2

# In[16]:


# Defining Parameters

r = 5 # Number of Rows
c = 5 # Number of Columns
gamma = 0.9 # Discount Factor

# Special States
A = (0,1)
newA = (4,1)

B = (0,3)
newB = (2,3)


# In[17]:


def isValidCell(a,b):
    return (0<=a and a<r and 0<=b and b<c)

def isEdgeCell(a,b):
    return ((a==0) or (a==r-1) or (b==0) or (b==c-1))

def isCornerCell(a,b):
    return((a==0 or a==r-1) and (b==0 or b==c-1))
    
def LocateNBR(a,b):
    NBR = []
    if(isValidCell(a-1,b)):
        NBR.append((a-1,b))
    if(isValidCell(a+1,b)):
        NBR.append((a+1,b))
    if(isValidCell(a,b-1)):
        NBR.append((a,b-1))
    if(isValidCell(a,b+1)):
        NBR.append((a,b+1))
        
    return NBR
    


# In[22]:


def BellmanEquation():
    P = np.zeros((r*c,r*c)) # Transition Probabilities
    R = np.zeros(r*c) # Reward Vector
    
    e = 0
    for i in range(r):
        for j in range(c):
            if((i,j) == A):
                P[e][c*newA[0] + newA[1]] = -gamma
                P[e][c*i + j] = 1
                R[e] = 10  # Reward from A to newA is 10
            elif((i,j) == B):
                P[e][c*newB[0] + newB[1]] = -gamma
                P[e][c*i + j] = 1
                R[e] = 5  # Reward from B to newB is 5
                
            else: # It is a normal cell
                if(isCornerCell(i,j)):
                    n = LocateNBR(i,j)
                    for k in n:
                        P[e][c*k[0] + k[1]] = -gamma/4 # Each direction is Equally probable
                    P[e][c*i + j] = (1-gamma/2)
                    R[e] = (-1/2)
                
                
                elif(isEdgeCell(i,j)):
                    n = LocateNBR(i,j)
                    for k in n:
                        P[e][c*k[0] + k[1]] = -gamma/4 # Each direction is Equally probable
                    P[e][c*i + j] = (1-gamma/4)
                    R[e] = (-1/4)
                    
                else:
                    n = LocateNBR(i,j)
                    for k in n:
                        P[e][c*k[0] + k[1]] = -gamma/4 # Each direction is Equally probable
                    P[e][c*i + j] = 1
                    R[e] = 0
            e+=1
                    
    return P, R


P, R = BellmanEquation()


# In[25]:


V = np.dot(np.linalg.inv(P),R)


# In[26]:


np.reshape(V,(r,c))


# ## Q4

# In[44]:


def OptimalPolicy():
    P = np.zeros((4*r*c, r*c))
    R = np.zeros(4*r*c)
    
    e = 0
    for i in range(r):
        for j in range(c):
            if((i,j) == A):
                P[e][c*newA[0] + newA[1]] = -gamma
                P[e][c*i + j] = 1
                R[e] = 10  # Reward from A to newA is 10
                e+=1
            elif((i,j) == B):
                P[e][c*newB[0] + newB[1]] = -gamma
                P[e][c*i + j] = 1
                R[e] = 5  # Reward from R to newR is 5
                e+=1
            else:
                if(isValidCell(i-1,j)):
                    P[e][c*(i-1) + j] = -gamma
                    P[e][c*i + j] = 1
                    R[e] = 0
                else:
                    P[e][c*i + j] = 1-gamma
                    R[e] = -1
                e+=1
                
                if(isValidCell(i+1,j)):
                    P[e][c*(i+1) + j] = -gamma
                    P[e][c*i + j] = 1
                    R[e] = 0
                else:
                    P[e][c*i + j] = 1-gamma
                    R[e] = -1
                e+=1
                
                if(isValidCell(i,j-1)):
                    P[e][c*i + (j-1)] = -gamma
                    P[e][c*i + j] = 1
                    R[e] = 0
                else:
                    P[e][c*i + j] = 1-gamma
                    R[e] = -1
                e+=1
                
                if(isValidCell(i,j+1)):
                    P[e][c*i + (j+1)] = -gamma
                    P[e][c*i + j] = 1
                    R[e] = 0
                else:
                    P[e][c*i + j] = 1-gamma
                    R[e] = -1
                e+=1
            
    return P,R


# In[54]:


P,R = OptimalPolicy()
tmp = np.ones(r*c)


vstar = np.round(np.reshape(optimize.linprog(tmp,-P,-R).x,(r,c)),1)


# In[87]:


qstar = [False]*(4*r*c)
#print(qstar)

for i in range(r):
    for j in range(c):
        print(i,j, end="")
        print(" - ",end="")
        
        if((i,j) == A or (i,j) == B):
            for k in range(4):
                qstar[4*(c*i+j) + k] = True
            print("UP,DOWN,LEFT,RIGHT")   
        else:
            check =[]
            if(isValidCell(i-1,j)):
                check.append(vstar[i-1][j])
            else:
                check.append(-1)
                
            if(isValidCell(i+1,j)):
                check.append(vstar[i+1][j])
            else:
                check.append(-1)
                
            if(isValidCell(i,j-1)):
                check.append(vstar[i][j-1])
            else:
                check.append(-1)
                
            if(isValidCell(i,j+1)):
                check.append(vstar[i][j+1])
            else:
                check.append(-1)
            
            check = np.array(check)
            d = np.where(check == np.max(check))
            for direc in d[0]:
#               qstar[4*(c*i+j) + direc] = True
                if(direc == 0):
                    print("UP,",end="")
                elif(direc == 1):
                    print("DOWN,",end="")
                elif(direc == 2):
                    print("LEFT,",end="")
                elif(direc == 3):
                    print("RIGHT",end="")
                
            print("")

