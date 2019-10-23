#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Mon Jun 17 15:16:40 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""
import os
import argparse

import numpy as np
import h5py as h5
import pickle
import shutil

import torch
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
from torch.utils import data

import OSToolBox as ost
import NN_builder_v2 as nb
import metrics as met
import data_reader as reader
import selecter as selec

import pandas as pd
from tqdm import tqdm

#import sys
#sys.path.append("/media/ostocker/Data/OS/git_clones/pytorch_modelsize")
#from pytorch_modelsize import SizeEstimator

#sys.path.append("/media/ostocker/Data/OS/git_clones/pytorch-receptive-field/")
#from torch_receptive_field import receptive_field


#SET of function to access h5 data
##########
def readH5Class(h5arg):
    with h5.File(h5arg, 'r') as f :
        return f.attrs["names"]
    
def readH5Channels(h5arg):
    with h5.File(h5arg, 'r') as f :
        return f.attrs["channels"]

def readH5Cweight(h5arg):
    with h5.File(h5arg, 'r') as f :
        return f.attrs["class_weight"]
##########


def learningPhase(args, model,loader, w, optimizer, metrics):
    #training function for one epoch of given loader
    model.train() # model in training mode
    
    #initialize metric container
    loss_meter = tnt.meter.AverageValueMeter()
    cm = met.ConfusionMatrix(len(args.c), args.c,args.nodata)
    
    #loop through batch given by data reader, unfold tuple and drop file name (the last param)
    for batch_ndx, (imgs,gt,__) in enumerate(tqdm(loader)):
        optimizer.zero_grad() #put gradient to zero
        
        #if GPU, load batch on it
        if args.cuda :
            batch_tensor = imgs.cuda()
        else :
            batch_tensor=imgs
        
        #generate prediction
        prediction = model(batch_tensor)
        prediction = prediction.cpu() #switch it on CPU for calculation
        
        #get & save batch loss and do backward
        loss = nn.functional.cross_entropy(prediction,gt,weight=w, ignore_index=args.nodata)
        loss.backward()
        loss_meter.add(loss.item())
        
        #clamp gradient to avoid some weight get high gradient actualization
        for p in model.parameters():
            p.grad.data.clamp(-1,1)
        
        #actualize weight
        optimizer.step()
        
        #calculate metrics if epoch_number % args.mem=0
        if metrics:
            for i in range(prediction.size()[0]):
                pred=prediction[i].argmax(0).squeeze()
                cm.add_batch(gt[i].numpy(), pred.numpy())
        
        #free memory
        del imgs
        del gt
        del batch_tensor
        del prediction
    
    return cm, loss_meter.value()[0]

def evalutionPhase(args, model,loader,w):
    #evaluation function for one epoch of given loader
    model.eval() # model in training mode
    
    #initialize metric container
    loss_meter = tnt.meter.AverageValueMeter()
    cm = met.ConfusionMatrix(len(args.c), args.c,args.nodata)
    
    #loop through batch given by data reader, unfold tuple and drop file name (the last param)
    for batch_ndx, (imgs,gt,__) in enumerate(tqdm(loader)):
        
        #if GPU, load batch on it
        if args.cuda :
            batch_tensor = imgs.cuda()
        else :
            batch_tensor=imgs
        
        #generate prediction
        prediction = model(batch_tensor)
        prediction = prediction.cpu() #switch it on CPU for calculation
        
        #get & save batch loss
        loss = nn.functional.cross_entropy(prediction,gt,weight=w, ignore_index=args.nodata)
        loss_meter.add(loss.item())
        
        #calculate metrics
        for i in range(prediction.size()[0]):
            pred=prediction[i].argmax(0).squeeze()
            cm.add_batch(gt[i].numpy(), pred.numpy())
        
        #free memory
        del imgs
        del gt
        del batch_tensor
        del prediction
    
    return cm, loss_meter.value()[0]


def save_checkpoint(state, is_best, directory):
    #function to save a checkpoint and overwrite best model if new best epoch
    torch.save(state, directory+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(directory+'/checkpoint.pth.tar', directory+'/model_best.pth.tar')


def full_train(args,cv,outIter):
    #function for one iteration of crossvalidation : training and evaluation of one model given cv data repartition
    print("Iteration file is ",outIter)
#    np.save(outIter+"/repartition_dalles.npy",cv) #saving data repartition of current CV iter
    
    #getting channels number to still match older version without channels data in meta_img
    try :
        channels=readH5Channels(args.meta_img)
        print("Image channels =", channels)
    except : 
        channels=4
        print("WARNING ! : OLD META_IMG : guessed 4 channels image")
    
    #initialize the model
    model=nb.OLIVENET(channels,len(args.c))
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    
    #load model on GPU if CUDA
    if args.cuda:
        model.cuda()
    
    #define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #Learning Rate decay //!\\ LAST ARGUMENT HARD CODED
    plan=[i for i in range(50,300,50)]
    gamma=0.7
    print('  |  '.join('{}: {:.2e}'.format(pl, lr) for pl, lr in zip(plan,[args.lr*gamma**i for i in range(len(plan))])))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=plan, gamma=gamma)
    
    #Case of pretrained model, load state
    if args.state is not False :
        print("TRANSFERT LEARNING : loading of :\n",args.state)
        checkpoint = torch.load(args.state)
        model.load_state_dict(checkpoint['state_dict'])
    
    #get the class weights from meta h5
    w=readH5Cweight(args.meta_gt)
    w=[1/np.sqrt(i) if i!=0 else 0 for i in w]
    w=w/np.linalg.norm(w)
    w=torch.from_numpy(w.astype(np.float32))
    print("Class weights:", [args.c[i]+": {:.4f} ".format(weight) for i,weight in enumerate(w)])
    
    #create the perf log saving class
    outLog=ost.createDir(outIter+"/perf_log")
    save_lr=[]
    perf=met.SavePerf(args.e, outLog,args.c) #PERF OBJECT : to add result be sure in which dataset you put the result in
    outState=ost.createDir(outIter+"/states")
    
    #saving data repartition of current CrossValidation iteration will be used by Drawer algorithm
    with open(outIter+"/crossValID.pckl",'wb') as fp:
        pickle.dump(cv[0],fp)
        pickle.dump(cv[1],fp)
        pickle.dump(cv[2],fp)
    
    #create data sets // the loader function will take care of the batching
    #suffle=True and Droplast=true only for train loader
    #Batch size * 3 because no gradient stored on GPU so more memory available for quicker inference
    train_loader=data.DataLoader(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt,cv[0], args.aug, (200,200)) \
                           , batch_size=args.bs,shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader=data.DataLoader(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt, cv[1]) \
                           , batch_size=args.bs*3,shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader=data.DataLoader(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt, cv[2]) \
                           , batch_size=args.bs*3,shuffle=False, drop_last=False, num_workers=args.num_workers)
    
    #epoch iterations [try and except KeyboardInterrupt] to allow ctrl+c interruption
    try:
        for i_epoch in range(args.e):
            #periodic metric calculation and model saving
            if i_epoch % args.mem == 0 or i_epoch == args.e-1: #if last epoch too
                print("epoch:",i_epoch, end='', flush=True)
                
                #train with train data and register train metrics
                cm, train_loss = learningPhase(args, model, train_loader, w, optimizer, True)
                perf.addFullResults(i_epoch,cm,train_loss,0)#0 for TRAIN dataset
                
                # evaluate and suppress gradient calcution for these steps
                with torch.no_grad():
                    #validation data and register validation metrics and get FLAG OF BEST EPOCH
                    cm, loss = evalutionPhase(args, model, val_loader, w)
                    is_best=perf.addFullResults(i_epoch,cm,loss,1)#0 for VAL dataset
                    
                    #test data and register test metrics [not required for train]
                    cm, loss = evalutionPhase(args, model, test_loader,w)
                    perf.addFullResults(i_epoch,cm,loss,2)#2 for TEST dataset
                
                #save model characteristics and states
                save_checkpoint({
                    'epoch' : i_epoch,
                    'channels' : channels,
                    'names' : args.c,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, is_best, outState)
            
            else :#just train without calculating metrics
                print("epoch:",i_epoch, end='', flush=True)
                __, train_loss= learningPhase(args, model, train_loader, w, optimizer, False)
                perf.addLossResults(i_epoch,train_loss,0)#0 for TRAIN dataset
                
            #increment learning rate step and save current value
            scheduler.step()
            save_lr.append([ group['lr'] for group in optimizer.param_groups ][0])
    except KeyboardInterrupt:
        pass
    
    #print performance results
    print("Best epoch:",perf.best)
    perf.printResults()#print graph
    perf.printResultsTxt()#print values in txt
    
    #plot learning rate graph and save values in txt
    ost.plotLogGraph(np.arange(len(save_lr)),save_lr,'lr','epochs','Lr','Learning rate decay according to epochs'\
                     ,outLog+'/lr_decay.tif',show=0)
    df=pd.DataFrame({'epochs':np.arange(len(save_lr)),'lr':save_lr})
    df.to_csv(outLog+"/lr_decay.csv", sep=";", index=False)
    
    return model


def crossTraining(args):
    #main function : manage cross validation iterations and plan its data repartition
    outAP=os.path.abspath(args.out)
    
    #load class names into args to be easily used everywhere
    args.c = readH5Class(args.meta_gt)
    
    #Planning of data repartition
    crossIndex=selec.selecter(args)
    iterartion = args.cvIter if args.cvIter<=args.cvSplit else args.cvSplit
    
    #print CV Planning
    print("Split: ", args.cvSplit, end="")
    if args.cvIter>=args.cvSplit : print(", Iter: ",args.cvIter," REDUCED TO ",iterartion)
    else : print(", Iter: ",args.cvIter)
    for i in range(iterartion):
        print("Iter "+str(i)+": train="+str(len(crossIndex[i][0]))+" val="+str(len(crossIndex[i][1])) \
              +" test="+str(len(crossIndex[i][2])))
    
    #cross validation loop
    for i in range(args.cvIter):
        outIter=ost.createDirIncremental(outAP+"/stepCrossVal", 0)
        model = full_train(args,crossIndex[i],outIter)
    
    return model


def mainProcess(args):
    #ULTRA_MAIN
    #cute but useless function
    trained_model=crossTraining(args)
    return trained_model



if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    #essentials
    parser.add_argument("-dir_img", type=str, help="directory of the img tiles", required=True)
    parser.add_argument("-dir_gt", type=str, help="directory of the gt tiles", required=True)
    parser.add_argument("-meta_img", type=str, help="h5 file of dataset metadata for img", required=True)
    parser.add_argument("-meta_gt", type=str, help="h5 file of dataset metadata for gt", required=True)
    parser.add_argument("-out", type=str, help="path: data saving path", default='', required=True)
    #path or false
    parser.add_argument("-state", type=ost.SFParser, help="path to model state .pth", default=False, required=False)
    parser.add_argument("-selection", type=ost.SFParser, help="directory of selection shapes", default=False, required=False)
    parser.add_argument("-seed", type=ost.SFParser, help="path to a numpy random state save", default=False, required=False)
    #values
    parser.add_argument("-nodata", type=int, help="no data value in images", default=99, required=False)
    parser.add_argument("-lr", type=float, help="float: learning rate", default=1e-2, required=False)
    parser.add_argument("-e", type=int, help="float: number of epoch", default=2, required=False)
    parser.add_argument("-bs", type=int, help="batch size", default=1, required=False)
    parser.add_argument("-cvIter", type=int, help="number of cross validation iteration", default=1, required=False)
    parser.add_argument("-cvSplit", type=int, help="number of cross validation split at least 3", default=5, required=False)
    parser.add_argument("-mem", type=int, help="set the number between each metrics epoch", default=2, required=False)
    parser.add_argument("-num_workers", type=int, help="number of dataloader workers", default=10, required=False)
    #boolean
    parser.add_argument("-cuda", type=ost.str2bool, help="true if graphic card", default=True, required=False)
    parser.add_argument("-aug", type=ost.str2bool, help="true data augmentation", default=False, required=False)
    #others
    parser.add_argument("-c", type=list, help="classes names leave empty", default=[], required=False)
    
    args = parser.parse_args()

    #main
    mainProcess(args)