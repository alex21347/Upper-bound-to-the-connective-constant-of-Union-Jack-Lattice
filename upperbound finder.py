#Alms Method of Finding Upper Bounds Applied to the Union Jack Lattice

#PLEASE NOTE : lines 78 and 91 in the code have been edited for 
#              display purposes, a simple re-edit will make the
#              code once again usable for general use

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm 

def take_a_step(current_walks):
    
    #This function enumarates the number self-avoiding walks of a given length
    
    new_valid_walks = np.array([[[]]]);
    for i in range(0,current_walks.shape[0]):
        if np.floor(current_walks[i,-1,0]) - current_walks[i,-1,0] == 0:
            steps = [[0,1],
                     [1,0],
                     [0,-1],
                     [-1,0],
                     [0.5,0.5],
                     [0.5,-0.5],
                     [-0.5,-0.5],
                     [-0.5,0.5]]
            for j in steps:
                new_pos_ij = np.array([current_walks[i,-1] + j])
                if np.any(np.all(current_walks[i,:] == new_pos_ij ,axis = 1) == True):
                    continue
                else:
                    new_walk_ij = np.append(current_walks[i,:], new_pos_ij,axis = 0)
                    new_walk_ij = np.array([new_walk_ij])
                    if new_valid_walks.shape[2] == 0:
                        new_valid_walks = new_walk_ij
                    else:
                       new_valid_walks = np.append(new_valid_walks, new_walk_ij,axis = 0)
        else:
            steps =  [[0.5,0.5],
                     [0.5,-0.5],
                     [-0.5,-0.5],
                     [-0.5,0.5]]
            
            for j in steps:
                new_pos_ij = np.array([current_walks[i,-1] + j])
                
                if np.any(np.all(current_walks[i,:] == new_pos_ij ,axis = 1) == True):
                    continue
                else:
                    new_walk_ij = np.append(current_walks[i,:], new_pos_ij,axis = 0)
                    new_walk_ij = np.array([new_walk_ij])
                    if new_valid_walks.shape[2] == 0:
                        new_valid_walks = new_walk_ij
                    else:
                        new_valid_walks = np.append(new_valid_walks, new_walk_ij,axis = 0)
    c_n = int(new_valid_walks.shape[0])
    return new_valid_walks,c_n


def generate_G(n,m,new_walks0,new_walks0_m,new_walks1,new_walks1_m, F_m):
    
   #This function generates the G matrix described in Alms Method
    
    num00 = int(c_n0[m-1])                  #number of m-walks starting at node 0
    num01 = int(c_n0[N-1])                  #number of N-walks starting at node 0
    num10 = int(c_n1[m-1])                  #number of m-walks starting at node 1
    num11 = int(c_n1[N-1])                  #number of N-walks starting at node 1
    num20 = int(c_n0[m-1] + c_n1[m-1])      #number of m-walks starting at either node
    G = np.zeros((num20,num20))
    
    
    #finding number of n-step walks starting with a particular m-step walk and 
    #ending with a translation of another m-step walk for a vertex class 0
    
    for k in range(0,num01):
        for j in range(0,num00):
            if np.all(new_walks0[k,:m+1,:] == new_walks0_m[j,:m+1,:]):
                for i in range(0,num20):
                    if np.all(new_walks0[k,N-m:,:]-new_walks0[k,N-m,:]+np.abs(np.floor(
                            new_walks0[k,N-m,:])-new_walks0[k,N-m,:])==F_m[i,:m+1,:]):
                        G[j,i] = G[j,i] + 1
                        break
                break
            
    #finding the number of n-step walks starting with a particular m-step walk and 
    #ending with a translation of another m-step walk for a vertex class 1
    
    for k in range(0,num11):
        for j in range(0,num10):
            if np.all(new_walks1[k,:m+1,:] == new_walks1_m[j,:m+1,:]):
                for i in range(0,num20):
                    if np.all(new_walks1[k,N-m:,:]-new_walks1[k,N-m,:]+np.abs(np.floor(
                            new_walks1[k,N-m,:])- new_walks1[k,N-m,:]) == F_m[i,:m+1,:]):
                        G[j+num00,i] = G[j+num00,i] + 1
                        break
                break
    return G 

def power_method(G,n):
    
    #This function utilises the power method of finding the largest 
    #Eigenvale of a matrix
    
     G = np.matrix(G)
     v_k = np.matrix(np.random.rand(G.shape[1]))
     for i in range(0,n):
         G_k = v_k*G
         alpha_k = G_k.max()
         v_k = G_k*(1/alpha_k)
     return alpha_k
                 
#%%
    
N_max = 5 #N-step walks to enumarate
M_max = 3 #M-step walks to enumarate

mu = np.zeros((N_max+1,M_max+1))
for b in tqdm(range(2,5)):     
    for c in range(1,3):        
        if c < b:
            N = b #N steps
            m = c
            new_walks0 = np.array([[[0,0]]])
            new_walks1 = np.array([[[0.5,0.5]]])
            c_n0 = np.zeros(N)
            c_n1 = np.zeros(N)
            
            for i in range(0,N):
                new_walks0,c_n0[i] = take_a_step(new_walks0)
                if i == m-1:
                    new_walks0_m = new_walks0
                    
            for i in range(0,N):
                new_walks1,c_n1[i] = take_a_step(new_walks1)
                if i == m-1:
                    new_walks1_m = new_walks1
                    
            F_m = np.append(new_walks0_m,new_walks1_m, axis = 0)

            G = generate_G(N,m,new_walks0,new_walks0_m,new_walks1,new_walks1_m, F_m)
            
            beta = 0 # shift coefficient for power method
            mu[N,m] = power_method(G - beta*np.identity(G.shape[1]),20)**(1/(N-m))