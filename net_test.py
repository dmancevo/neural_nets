# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:54:51 2015

@author: Diego
"""

import numpy as np
import matplotlib.pyplot as plt
from nlp_net import word_vec, conv_layer, pool_layer, layer, hinge_loss

def test_word_conv_pool():
    l1 = word_vec(word_dim=2)
    l2 = conv_layer(word_dim=2, K=(4,0), lrate=1.0, c=20)
    l3 = pool_layer()
    l4 = layer(dim=(4,1),lrate=1.0,f=lambda x: x,df=lambda x: 1,
               drp_rate=0.0, c=30)
    l5 = layer(dim=(1,1),lrate=1.0,f=np.tanh, df=lambda x: (1-np.square(x)),
               drp_rate=0.5, c=30)
               
    h = hinge_loss(m=1)
    
    positive_words = ['happy','fantastic','great','awesome',':)','faboulous','animated',
                   'excited','bombastic','amazing','good']
                   
    negative_words = ['sad','terrible', 'aweful', 'grief',':(','malign',
                        'depressed', 'tear', 'cry']
                        
    neutral_words = ['neutral', 'white', 'normal', 'jump', 'water']
    
    def f(words1, words2):                
        for _ in range(5):
            for p_word in words1:
                for n_word in words2:
                    
                    #Forward pass
                    z1_p, p_word = l1.forward(p_word)
                    z2_p = l2.forward(z1_p)
                    z3_p, z3_ind_max_p = l3.forward(z2_p)
                    z4_p, v4_p, drp4_p = l4.forward(z3_p)
                    z5_p, v5_p, drp5_p = l5.forward(z4_p)
                    
                    z1_n, n_word = l1.forward(n_word)
                    z2_n = l2.forward(z1_n)
                    z3_n, z3_ind_max_n = l3.forward(z2_n)
                    z4_n, v4_n, drp4_n = l4.forward(z3_n)
                    z5_n, v5_n, drp5_n = l5.forward(z4_n)
        
                    
                    #Hinge loss
                    Z = h.forward(z4_p[0], z4_n[0])
                    dE_dZ = h.backward(Z)
                    
                    #Backward deltas
                    d1_p = l5.backward([dE_dZ[0]], z5_p, drp5_p)
                    d2_p = l4.backward([dE_dZ[0]], z4_p, drp4_p)
                    d3_p = l3.backward(d2_p, z3_ind_max_p, z2_p)
                    d4_p = l2.backward(d3_p,z1_p)
                    
                    d1_n = l5.backward([dE_dZ[1]], z5_n, drp5_n)
                    d2_n = l4.backward([dE_dZ[1]], z4_n, drp4_n)
                    d3_n = l3.backward(d2_n, z3_ind_max_n, z2_n)
                    d4_n = l2.backward(d3_n,z1_n)
                    
                    #Update learned parameters
                    l5.update([dE_dZ[0]], z5_p, v5_p)
                    l4.update([dE_dZ[0]], z4_p, v4_p)
                    l2.update(d3_p, z1_p)
                    l1.update(d4_p, p_word)
                    
                    l5.update([dE_dZ[1]], z5_n, v5_n)
                    l4.update([dE_dZ[1]], z4_n, v4_n)
                    l2.update(d3_n, z1_n)
                    l1.update(d4_n, n_word)
                    
    def plot_word_vecs(words, color):
        Px, Py = [l1.W[word][0] for word in words], [l1.W[word][1] for word in words]
        plt.scatter(Px,Py, color=color)
            
    f(positive_words, negative_words)
    f(positive_words, neutral_words)
    f(negative_words, neutral_words)
    
    plot_word_vecs(positive_words, 'blue')
    plot_word_vecs(negative_words, 'red')
    plot_word_vecs(neutral_words, 'green')

def test_layer():

    X, Y = [], []
    
    for i in range(1000):
        
        x = np.random.uniform(-3,3)
        
        X.append(np.array([x]))
        Y.append(-np.sin(x)+np.random.normal(0,0.1))
        
    X2, Y2 = [], []
    
    for i in range(1000):
        
        x = np.random.uniform(-3,3)
        
        X2.append(np.array([x]))
        Y2.append(-np.sin(x)+np.random.normal(0,0.1))
        
        
    
    l1 = layer(dim=(1,8), f=np.tanh, df=lambda x: (1.0-np.square(x)),
    drp_rate=0.0, lrate=3.0, c=1)
    
    l2 = layer(dim=(8,1), f=lambda x: x, df=lambda x: 1.0,
    drp_rate=0.5, lrate=3.0, c=1)
    
    Y_hat = []
    
    for j in range(len(X)):
        z1, v1, drp1 = l1.forward(X[j])
        z2, v2, drp2 = l2.forward(z1)
        
        dE_dZ = z2-Y[j]
        
        d1 = l2.backward(dE_dZ, z2, drp2)
        
        l1.update(d1, z1, v1)
        l2.update(dE_dZ, z2, v2)
        
    for j in range(len(X2)):
        z1 = l1.forward(X2[j], train=False)
        z2 = l2.forward(z1, train=False)
        Y_hat.append(z2)
    
    plt.scatter([x2[0] for x2 in X2],Y2, color='blue')
    plt.scatter([x2[0] for x2 in X2],Y_hat, color='red')
    
test_layer()
test_word_conv_pool()
    
