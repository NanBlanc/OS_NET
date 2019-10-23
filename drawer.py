#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Thu Jun 27 15:00:51 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""
import argparse
import pickle

import sys

import OSToolBox as ost
import NN_builder as nb
import metrics as m
import data_reader as reader

import h5py as h5
import numpy as np
from osgeo import gdal

import torch
import torch.nn as nn
from torch.utils import data
import multiprocessing 

from tqdm import tqdm


def multiprocessing_func(mpArg):
    #function to produce required inference product for one tile
    
    #extract arguments
    prediction=mpArg[0]
    gt=mpArg[1]
    base_name=mpArg[2]
    outs=mpArg[3]
    args=mpArg[4]
    colors=mpArg[5]
    
    #take arg max of prediction tensor for dif and inf product 
    if args.inf or args.dif:
        pred=prediction.argmax(0).squeeze().numpy()
        #cut prediction if margin
        if args.margin!=0:
            pred=pred[args.margin:pred.shape[0]-args.margin,args.margin:pred.shape[1]-args.margin]
            gt=gt[args.margin:gt.shape[0]-args.margin,args.margin:gt.shape[1]-args.margin]
    
    #produce semantic segmentation map
    if args.inf :
        path=outs[0]+"/inf_"+ost.pathLeaf(base_name)+".tif"
        ost.array2raster(pred, path, base_name,rasterType='GTiff', datatype=gdal.GDT_Byte, noDataValue=args.nodata, colors=colors,margin=args.margin)
    
    #produce difference map between GT and prediction arg max map
    if args.dif :
        colors_dif={0:(0,0,0,0),1:(255,0,0,255)}
        dif=ost.arrayDif(gt,pred,args.nodata)
        path=outs[1]+"/dif_"+ost.pathLeaf(base_name)+".tif"
        ost.array2raster(dif, path, base_name,rasterType='GTiff', datatype=gdal.GDT_Byte, noDataValue=0, colors=colors_dif,margin=args.margin)
    
    #produce proba maps (1 image for each class)
    if args.proba : 
        #convert logits to probabilities with softmax
        pred_soft=nn.functional.softmax(prediction,0)
        #iterate on each class (output dir)
        for i,o in enumerate(outs[2]):
            path=o+"/proba_"+ost.pathLeaf(base_name)+".tif"
            c=pred_soft[i].numpy()
            #cut prediction if margin
            if args.margin!=0:
                c=c[args.margin:c.shape[0]-args.margin,args.margin:c.shape[1]-args.margin]
            ost.array2raster(c, path, base_name,rasterType='GTiff', datatype=gdal.GDT_Float32,margin=args.margin)
    
    #produce proba maps in h5 format (saved as full 3D tensor)
    if args.probaH5:
        path=outs[3]+"/proba_"+ost.pathLeaf(base_name)+".h5"
#            print(path)
        with h5.File(path, "w") as f:
            c=prediction.numpy()
            #cut margin and save
            f.create_dataset("pred",data=c[:,args.margin:c.shape[1]-args.margin,args.margin:c.shape[2]-args.margin])



def inference(ind, model, loader, args):
    #function to produce inference data for one data loader
    #ind : loop position
    #model : explicit
    #loader : current data loader
    #args : look at parser args
    
    #create all subfolder if train/val/test split, else don't create new subfolders
    outs=["","",[],[]]
    name=["train","val","test"] #need to match the order of dataloader
    if args.inf:
        outs[0]=ost.createDir(args.out+"/inf/"+name[ind]) if args.train_set else args.out+"/inf"
    if args.dif:
        outs[1]=ost.createDir(args.out+"/dif/"+name[ind]) if args.train_set else args.out+"/dif"
    if args.proba:
        outHM = [ost.createDir(args.out+"/proba/"+c) for c in args.c]
        outs[2] = [ost.createDir(p+"/"+name[ind]) for p in outHM] if args.train_set else outHM
    if args.probaH5:
        outs[3]=ost.createDir(args.out+"/probaH5/"+name[ind]) if args.train_set else args.out+"/probaH5"
#        print(outs[3])
    
    #get colors as dict for colortable
    if args.color is not False:
        c=np.loadtxt(args.color,delimiter=",",dtype=int)
        colors={}
        for i in range(c.shape[0]):
            colors[int(c[i][0])]=(tuple(c[i][1:5]))
    else :
        colors =None
    
    #metric containers
    if args.metric:
        cm = m.ConfusionMatrix(len(args.c), args.c,args.nodata)
    
    #loop through batch given by data reader, unfold tuple
    for batch_ndx, (imgs,gt,names) in enumerate(tqdm(loader)):
        
        #if CUDA, load on GPU
        if args.cuda :
            batch_tensor = imgs.cuda()
        else :
            batch_tensor=imgs
        
        #generate prediction
        prediction = model(batch_tensor)
        prediction = prediction.cpu()
        
        #build instruction for multiprocess loop (1 instruction for each tile of the batch)
        if args.noGT:
            mpArg=[(prediction[i].detach(),None,names[i],outs,args,colors) for i in range(prediction.shape[0])]
        else :
            mpArg=[(prediction[i].detach(),gt[i],names[i],outs,args,colors) for i in range(prediction.shape[0])]
        
        #multi process loop to create inference data of the current batch
        with multiprocessing.Pool() as pool:
            pool.map(multiprocessing_func, mpArg)
            
        #calculate batch metric and add it to metric containers
        if args.metric:
            for i in range(prediction.size()[0]):
                pred=prediction[i].argmax(0).squeeze()
                cm.add_batch(gt[i].numpy(), pred.numpy())
        
        #free memory
        del imgs
        del gt
        del batch_tensor
        del prediction
    
    #produce current dataset metric to metric output folder
    if args.metric:
        out_perf=ost.createDir(args.out+"/"+name[ind]+"_perf_inf") if args.train_set else ost.createDir(args.out+"/perf_inf")
        cm.printPerf(out_perf)
    return 0


def main(args):
    #MAIN FUNCTION FOR ARGS LOOK AT PARSER
    #RETURN : NOTHING ! HA ! but create all inference data in output dir
    
    #manage unmatching args (there may exist cleverer solution, it was implemented to bu simple)
    if not args.inf and not args.dif and not args.proba and not args.probaH5:
        print("ERROR : WAKE UP !! ... select at least one things to do (inf/dif/proba/probaH5)")
        return 0
    if args.dir_gt is False and args.meta_gt is False:
        args.noGT=True #if no gt noGT is true
    elif args.dir_gt is not False and args.meta_gt is False :
        print("ERROR : Either set -dir_gt to False or give meta_gt path")
        return 0
    elif (args.dir_gt is False and args.meta_gt is not False):
        print("ERROR : Either set -meta_gt to False or give dir_gt path")
        return 0
    else :
        args.noGT=False
    if args.noGT and args.dif:
        print("ERROR : You can't get dif between Inf and GT without GT ... Either set -noGT to False or -dif to False")
        return 0
    if args.noGT and args.metric:
        print("ERROR : You can't get metric without GT ... Either set -noGT to False or -metric to False")
        return 0
    
    #create output dir
    args.out=ost.createDir(args.out+"/drawings")
    
    #if data repartition file given, create separate train/val/test dataset reader
    DS_list=[]
    if args.cv is not False:
        args.train_set=True
        with open(args.cv,'rb') as fp:
            train = pickle.load(fp)
            val = pickle.load(fp)
            test = pickle.load(fp)
        #create data sets 
        DS_list.append(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt, train, noGT=args.noGT))
        DS_list.append(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt,val, noGT=args.noGT))
        DS_list.append(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt,test, noGT=args.noGT))
    else:
        DS_list.append(reader.DatasetTIF(args.dir_img, args.dir_gt, args.meta_img, args.meta_gt, noGT=args.noGT))
    
    #create list of data loader given created data reader
    loader_list=[data.DataLoader(ds, batch_size=args.bs,shuffle=False, drop_last=False, 
                                 num_workers=args.num_workers) for ds in DS_list]

    #load model
    checkpoint = torch.load(args.state)
    print("Best epoch: ",checkpoint['epoch'])
    try :
        channels=checkpoint['channels']
        args.c=checkpoint['names']
        print("Model channels :", channels,"\nModel class :", args.c)
    except :
        print("ERROR : OLD MODEL, PLEASE ACTUALIZE")
        sys.exit()
    model=nb.OLIVENET(channels,len(args.c))
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval() # model in training mode
    
    #create output dir regarding to asked operations
    if args.inf:
        ost.createDir(args.out+"/inf")
    if args.dif:
        ost.createDir(args.out+"/dif")
    if args.proba:
        ost.createDir(args.out+"/proba")
    if args.probaH5:
        ost.createDir(args.out+"/probaH5")
    
    #inference loop through all data loader created
    for i,loader in enumerate(loader_list):
        inference(i, model, loader, args)


if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    #essentials
    parser.add_argument("-state", type=str, help="path to model file .pth", required=True)
    parser.add_argument("-meta_img", type=str, help="h5 file of dataset metadata for img", required=True)
    parser.add_argument("-dir_img", type=str, help="path to directory containing images", required=True)
    parser.add_argument("-out", type=str, help="output dir", required=True)
    #path or false
    parser.add_argument("-meta_gt", type=ost.SFParser, help="h5 file of dataset metadata for gt", default=False, required=False)
    parser.add_argument("-dir_gt", type=ost.SFParser, help="path to directory containing gt", default=False, required=False)
    parser.add_argument("-color", type=ost.SFParser, help="path to color txt file as value,r,g,b,a per line for each value to color", default=False, required=False)
    parser.add_argument("-cv", type=ost.SFParser, help="pickle of index for train/test/val needed if train_set=true", default=False, required=False)
    #booleans
    parser.add_argument("-metric", type=ost.str2bool, help="set to true to get metric", default=True, required=False)
    parser.add_argument("-inf", type=ost.str2bool, help="if out classification as tif",default=True, required=False)
    parser.add_argument("-dif", type=ost.str2bool, help="if out dif between gt as tif",default=False, required=False)
    parser.add_argument("-proba", type=ost.str2bool, help="if ou proba as tif",default=False, required=False)
    parser.add_argument("-probaH5", type=ost.str2bool, help="if out h5 file with proba",default=False, required=False)
    parser.add_argument("-cuda", type=ost.str2bool, help="true if graphic card", default=True, required=False)
    #values
    parser.add_argument("-margin", type=int, help="margin took for image overlap", default=0, required=False)
    parser.add_argument("-num_workers", type=int, help="number of dataloader workers", default=10, required=False)
    parser.add_argument("-bs", type=int, help="batch size", default=1, required=False)
    parser.add_argument("-nodata", type=int, help="no data value in images", default=99, required=False)
    #others
    parser.add_argument("-train_set", type=bool, help="bool LEAVE EMPTY", default=False, required=False)
    parser.add_argument("-noGT", type=bool, help="bool LEAVE EMPTY", default=False, required=False)
    parser.add_argument("-c", type=list, help="classes names - LEAVE EMPTY", default=[], required=False)
    
    args = parser.parse_args()
    
    #main
    main(args)
