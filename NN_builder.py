#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Tue May 28 10:40:24 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""
import torch
import torch.nn as nn


class OLIVENET(nn.Module):
#model class
    def __init__(self,nb_channels,nb_class):
    #is build modular on channel and class number => theses param need to be saved with model
        super(OLIVENET,self).__init__()
        print("Model_v2")
        
        #modules without parameters (can be used multiple times)
        self.mp=nn.MaxPool2d(2,2,return_indices=True)
        self.rl= nn.ReLU()
        self.ump=nn.MaxUnpool2d(2,2)
        
        #modules with parameters (can be used only once)
        #encoder
        self.c1= nn.Sequential(nn.Conv2d(nb_channels,64,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(64))
        self.c2=nn.Sequential(nn.Conv2d(64,64,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(64))
        
        self.c3=nn.Sequential(nn.Conv2d(64,128,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(128))
        self.c4=nn.Sequential(nn.Conv2d(128,128,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(128))
        
        self.c5=nn.Sequential(nn.Conv2d(128,256,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(256))
        #decoder
        self.c6=nn.Sequential(nn.Conv2d(256,128,3,padding=1, padding_mode='reflection'),nn.BatchNorm2d(128))
        
        self.dc5=nn.Sequential(nn.ConvTranspose2d(256,128,3,padding=1),nn.BatchNorm2d(128))
        self.dc6=nn.Sequential(nn.ConvTranspose2d(128,64,3,padding=1),nn.BatchNorm2d(64))
        
        self.dc7=nn.Sequential(nn.ConvTranspose2d(128,64,3,padding=1),nn.BatchNorm2d(64))
        self.dc8=nn.Sequential(nn.ConvTranspose2d(64,64,3,padding=1),nn.BatchNorm2d(64))
        #final condensing layer
        self.dc9=nn.ConvTranspose2d(64,nb_class,3,padding=1)
        
        #weight nitialization
        self.c1.apply(self.init_weights)
        self.c2.apply(self.init_weights)
        self.c3.apply(self.init_weights)
        self.c4.apply(self.init_weights)
        self.c5.apply(self.init_weights)
        self.c6.apply(self.init_weights)
        
        self.dc5.apply(self.init_weights)
        self.dc6.apply(self.init_weights)
        self.dc7.apply(self.init_weights)
        self.dc8.apply(self.init_weights)
        self.dc9.apply(self.init_weights)
    
    
    def init_weights(self,m):
        if type(m) == nn.Conv2d or type(m)==nn.ConvTranspose2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            nn.init.xavier_uniform_(m.weight) #old init
    
    def forward(self,x0):
#        print(x0[0],x0[1])
#        print('etage 0')
        #x0=self.rl(self.m1(self.c1(x0)))
        x0=self.rl(self.c1(x0))
        x0=self.rl(self.c2(x0))
#        print(x0.size())
        
#        print('etage 1')
        x1,indices01=self.mp(x0)# indices=maxpool indices needed for unpooling step
        x1=self.rl(self.c3(x1))
        x1=self.rl(self.c4(x1))
#        print(x1.size())
        
#        print('etage 2')
        x2,indices12=self.mp(x1)
        x2=self.rl(self.c5(x2))
        x2=self.rl(self.c6(x2))
#        print(x2.size())
        
#        print('etage -1')
        x=self.ump(x2,indices12,x1.size())
        x=torch.cat((x1,x),1)#concatenation operation for pseudo residual connexion : may be switched to addition 
        x=self.rl(self.dc5(x))
        x=self.rl(self.dc6(x))
#        print(x.size())1
        
#        print('etage -0')
        x=self.ump(x,indices01,x0.size())
        x=torch.cat((x0,x),1)
        x=self.rl(self.dc7(x))
        x=self.rl(self.dc8(x))
        #logits=self.rl(self.dc9(x))
        logits=(self.dc9(x))
#        print(logits.size())
        return logits