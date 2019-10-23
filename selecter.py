#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Thu Jul 11 15:32:58 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""


import OSToolBox as ost
from osgeo import ogr
import argparse
import numpy as np
import pickle
import itertools
import h5py as h5

def readH5Length(h5arg):
    with h5.File(h5arg, 'r') as f :
        return f.attrs["len"]

def paquetSpliter(full_index, nb_split):
    #Take a list of index and split it into "nb_split" sub lists
    #distribute division reminder if len(full_index)%nb_split!=0
    #RETURN list of list
    
    all_sub=[]
    paquet=len(full_index)//nb_split
    reste=len(full_index)%nb_split
    decalage=0
    for i in range(nb_split):
        if reste!=0:
            repartition=1
            reste-=1
        else:
            repartition=0
        sub=full_index[i*paquet+decalage:(i+1)*paquet+repartition+decalage]
        if repartition==1:
            decalage+=1
        all_sub.append(sub)
    return all_sub

def crossValSpliter(prng,data_nb,split, check=True):
    #PRNG : random seed
    #data_nb: full dataset lenght
    #split : number of CV iteration
    #check : verify if no redundance of tiles couples in all CV iter and thus data leaking
    #RETURN : [list of list of int] cross validation iteration list of dataset (train/val/test) list of index of tile couple
    #But don't manage locked tiles
    
    #randomize order
    perm=prng.permutation(data_nb)
    
    #split all_index into "split" sub lists of approximative same lenght
    all_sub=paquetSpliter(perm,split)
    
    #construct all iterations by iteratively merging "split"-2 sublists for train and keeping 1 for val and 1 for test
    cross=[]
    for i in range(split):
        im=[] #index modulo to iterate over
        for j in range(split): 
            im.append((i+j)%split)
        test=all_sub[im[0]]
        val=all_sub[im[1]]
        train=[]
        for a in im[2:split]:
            train=train + list(all_sub[a])
        cross.append((train,val,test))
    
    #check CV
    if check:
        #check
        compteur_test=np.zeros(perm.shape)
        compteur_train=np.zeros(perm.shape)
        compteur_val=np.zeros(perm.shape)
#       print("paquets: ", all_sub)
        for i in range(len(cross)):
#           print(str(i)," : \ntrain: ", cross[i][0],"\nval: ",cross[i][1],"\ntest: ",cross[i][2])
            for j in range(len(cross[i][2])):
                compteur_test[cross[i][2][j]]+=1
            for j in range(len(cross[i][0])):
                compteur_train[cross[i][0][j]]+=1
            for j in range(len(cross[i][1])):
                compteur_val[cross[i][1][j]]+=1
        #display
        print("\nChecking cross validation:")
        print("\tTrain set:",np.sum(compteur_train)/len(compteur_train)==split-2) #only ones
        print("\tVal set:  ",np.sum(compteur_val)/len(compteur_val)==1)
        print("\tTest set: ",np.sum(compteur_test)/len(compteur_test)==1)
    
    return cross

def crossValSpliterSelection(prng,data_nb,split,selected,check=True):
    #PRNG : random seed
    #data_nb: full dataset lenght
    #split : number of CV iteration
    #selected : [list of list of int] locked tiles couples in each dataset (train/val/test)
    #check : verify if no redundance of tiles couples in all CV iter and thus data leaking
    #RETURN : [list of list of int] cross validation iteration list of dataset (train/val/test) list of index of tile couple
    #manage locked tiles
    
    #extract locked tiles from set
    all_data=np.arange(data_nb)
    selected_flat=list(itertools.chain(*selected))
    not_selected=np.delete(all_data,selected_flat)
    
    #create CV without locked tiles
    cross_not_selected_id=crossValSpliter(prng,len(not_selected),split,check)#work as index so that it can be from 0 to i with cuts
    
    #add locked tiles in respective dataset
    cross=[]
    for iter_cross_not_selected_id in cross_not_selected_id:
        iter_cross=[]
        for i in range(3):
            iter_cross_dt=[not_selected[idx] for idx in iter_cross_not_selected_id[i]]
            iter_cross_dt=iter_cross_dt+selected[i]
            iter_cross.append(iter_cross_dt)
        cross.append(iter_cross)
    
    #check CV with locked tiles
    if check:
        #check
        compteur_test=np.zeros(data_nb)
        compteur_train=np.zeros(data_nb)
        compteur_val=np.zeros(data_nb)
        for i in range(len(cross)):
    #        print(str(i)," : \ntrain: ", cross[i][0],"\nval: ",cross[i][1],"\ntest: ",cross[i][2])
            for j in range(len(cross[i][2])):#test
                compteur_test[cross[i][2][j]]+=1
            for j in range(len(cross[i][0])):#train
                compteur_train[cross[i][0][j]]+=1
            for j in range(len(cross[i][1])):#val
                compteur_val[cross[i][1][j]]+=1
        
        #display
        print("\nChecking cross validation with selection :")
        print("\tTrain set:",np.sum(compteur_train)==((split-2)*len(not_selected)+split*len(selected[0])))
        print("\tVal set:  ",np.sum(compteur_val)==(len(not_selected)+split*len(selected[1])))
        print("\tTest set: ",np.sum(compteur_test)==(len(not_selected)+split*len(selected[2])))
    
    return cross


def selecter(args):
    #MAIN FUNCTION FOR ARGS LOOK AT PARSER
    #RETURN : [list of list of int] cross validation iteration list of dataset (train/val/test) list of index of tile couple
    
    #order = test,train,val by alphabetic order
    order_good=[2,3,1] #to read shapefiles in train/val/test order || 0 is all_index and is skipped
    data_lenght=readH5Length(args.meta_gt)
    selected=[[] for i in order_good] # init container of container of locked tiles
    
    #read shapefiles to see if some tiles couple will be locked some dataset all along CV
    try:
        shps=ost.getFileByExt(args.selection,"shp") #get shapefiles
        print("\nReading selection shapefiles:")
        for i,order in enumerate(order_good):
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataSource = driver.Open(shps[order], 0)
            print("\t",shps[order])
            layer = dataSource.GetLayer()
            featureCount = layer.GetFeatureCount()
            if featureCount<data_lenght: #if files are full, ignore them
                for feature in layer:
                    name=ost.pathLeaf(feature.GetField("location")) # retrieve tile name
                    #extract tile number
                    index=0
                    tile_num=None
                    while tile_num is None :
                        try :
                            tile_num=int(name.split("_")[index])
#                            print(index)
                        except:
                            index+=1
                            pass
                    if tile_num is None :
                        print("\tWARNING/!\: Error in tile numerotation, was not able to get the tile number")
                    else :
                        selected[i].append(tile_num)
            layer.ResetReading() # assure reset layer reading data variable
        
        print("\nSelection found :\n\tTrain samples:",len(selected[0]),"\n\tVal samples:  ",len(selected[1]),"\n\tTest samples: ",len(selected[2]))
    except:
        print("\tWARNING/!\: Path to data selection shapefiles directory not found")
        pass
    
    
    #load/set seed for random data repartition [if new seed, will be saved]
    if args.seed is False :
        seedsave=ost.checkNIncrementLeaf(args.out+"/seed.dump")
        with open(seedsave,'wb') as fp:
            prng = np.random.RandomState()
            pickle.dump(prng,fp)
    else:
        with open(args.seed,'rb') as fp:
            prng = pickle.load(fp)
    
    #cross validation creation
    cross= crossValSpliterSelection(prng,data_lenght,args.cvSplit,selected)
#    print(cross)
    return cross


if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-selection", type=str, help="directory of selection shapes", required=True)
    parser.add_argument("-meta_gt", type=str, help="h5 file of dataset metadata", required=True)
    parser.add_argument("-cvSplit", type=int, help="number of cross validation split at least 3", default=5, required=False)
    parser.add_argument("-seed", type=str, help="path to a numpy random state save", required=False)
    parser.add_argument("-out", type=str, help="dir where to save the seed", default='',required=False)

    args = parser.parse_args()

    #main
    selecter(args)