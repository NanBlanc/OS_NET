#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Tue Jun 18 15:34:38 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""

import torch
from torch.utils import data
import h5py as h5
import numpy as np
from osgeo import gdal_array
#import OSToolBox as ost

class DatasetTIF(data.Dataset): 
    #personnal pytorch data loader to load each GT / img tile couple and perform data augmentation on it
    def __init__(self, dir_img,dir_gt=None,meta_img=None,meta_gt=None, id_list=[], transform = False, cropSize=(0,0), noGT=False):
        #dir_img : img tile folder
        #dir_img : gt tile folder
        #meta_img: meta data h5 img file
        #meta_gt: meta data h5 gt file
        #id_list: tile indice of current data set
        #transform : True if data augmentation
        #CropSize : size of final croped images when data augmentation
        #noGT : SET TO TRUE IF YOU DIDNT GIVE GT
        
        self.noGT=noGT
        self.dir_img=dir_img
        with h5.File(meta_img, 'r') as f :
            self.img_names=f["tile_path"][:]
            #if no id list given : consider all tiles in this dataset
            self.id_list=np.arange(f.attrs["len"]) if len(id_list)==0 else id_list
        if not noGT :
            self.dir_gt=dir_gt
            with h5.File(meta_gt, 'r') as f :
                self.gt_names=f["gt_path"][:]
        self.transform=transform
        self.cropSize=cropSize
    
    
    def loadTifInstance(self,path):
            return gdal_array.LoadFile(path)
    
    def loadH5Att(self,f,path,att):
            return f[path].attrs[att]
    
    def getID(self,index):
        return self.id_list[index]
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.id_list)
    
    def __getitem__(self, index):
        #MAIN GETITEM FUNCTION : MANAGE CASE WHERE NO GT
        if self.noGT :
            return self.getitemNoGT(index)
        else:
            return self.getitem(index)
    
    def getitem(self, index):
        'Generates one sample of data in case of GT given'
        # Select sample
        ID = self.getID(index)
        img_f=self.dir_img+"/"+self.img_names[ID]
        gt_f=self.dir_gt+"/"+self.gt_names[ID]
        
        # Load img
        img = self.loadTifInstance(img_f)
        if len(img.shape)<3:#if image is only one channel : expand in channel dim to not confuse pytorch (to always have a 3D tensor)
            img=np.expand_dims(img,0)
        img = torch.from_numpy(img.astype(np.float32))
        
        #load GT
        label = self.loadTifInstance(gt_f)
        label = torch.from_numpy(label).long()
        
        #data augmentation
        if self.transform :
            #random crop
            size1=self.cropSize[0]
            size2=self.cropSize[1]
            start1=np.random.randint(0,img.shape[1]-size1+1)#randomize y position
            start2=np.random.randint(0,img.shape[2]-size2+1)#randomize x position
            img=img[:,start1:start1+size1,start2:start2+size2]#crop second and third dim for img (because first is channel)
            label=label[start1:start1+size1,start2:start2+size2]#crop first and second dim for gt
            
            #hflip and vflip
            if np.random.random() < 0.5: #1 chance over 2 
                img=torch.flip(img, [2])
                label=torch.flip(label, [1])
            if np.random.random() < 0.5:
                img=torch.flip(img, [1])
                label=torch.flip(label, [0])
            
            #90° deg rotation
            rot_nb=np.random.randint(0,4)
            img=torch.rot90(img, rot_nb, [1, 2])
            label=torch.rot90(label, rot_nb, [0, 1])
        
        return img, label,img_f
    
    def getitemNoGT(self, index):
        'Generates one sample of data in case of NOOO GT given'
        # Select sample
        ID = self.getID(index)
        img_f=self.dir_img+"/"+self.img_names[ID]
        
        # Load img 
        img = self.loadTifInstance(img_f)
        img = torch.from_numpy(img.astype(np.float32))
        
        #data augmentation
        if self.transform :
            #random crop
            size1=self.cropSize[0]
            size2=self.cropSize[1]
            start1=np.random.randint(0,img.shape[1]-size1+1)
            start2=np.random.randint(0,img.shape[2]-size2+1)
            img=img[:,start1:start1+size1,start2:start2+size2]
            
            #hflip and vflip
            if np.random.random() < 0.5:
                img=torch.flip(img, [2])
            if np.random.random() < 0.5:
                img=torch.flip(img, [1])
            
            #90° deg rotation
            rot_nb=np.random.randint(0,4)
            img=torch.rot90(img, rot_nb, [1, 2])
        return img,0, img_f



#####OTHER OLD LOADER USED in V1 & V2 => uncomment to reactivate

#class DatasetRAM(data.Dataset):
#    def __init__(self, in_h5, in_id_list, cuda = True):
#        self.hFile=in_h5
#        self.id_list=in_id_list
#        self.cuda=cuda
#        self.loadH5()
#    
#    def loadH5(self):
#        self.imgs=[]
#        self.gt=[]
#        grp_list=["img","gt"]
#        with h5.File(self.hFile, 'r') as f :
#            for name in grp_list:
#                grp = f[name]
#                for ID in self.id_list:
##                    print("ID:",ID)
#                    if name=="img":
#                        d=grp[str(ID)][:]
#                        self.imgs.append(torch.from_numpy(np.transpose(d.astype(np.float32),(2,0,1))))
#                    else :
#                        d=grp[str(ID)][:]
#                        self.gt.append(torch.from_numpy(d).long())
#            return 0
#        
#    def getID(self,index):
#        return self.id_list[index]
#    
#    def __len__(self):
#        'Denotes the total number of samples'
#        return len(self.id_list)
#    
#    def __getitem__(self, index):
#        'Generates one sample of data'
##        print(index)
#        # Load img and get label
#        img = self.imgs[index]
##        if self.cuda :
##            img = img.cuda()
#        label = self.gt[index]
#        
#        return img, label




#class Dataset(data.Dataset):
#    def __init__(self, hFile, id_list=[], transform = False, cropSize=(0,0)):
#        self.hFile=hFile
#        if len(id_list)==0:
#            with h5.File(hFile, 'r') as f :
#                self.id_list=np.arange(f.attrs["len"])
#        else :
#            self.id_list=id_list
#        self.transform=transform
#        self.cropSize=cropSize
#    
#    def loadH5Instance(self,f,path):
#            return f[path][:]
#    
#    def loadH5Att(self,f,path,att):
#            return f[path].attrs[att]
#    
#    def getID(self,index):
#        return self.id_list[index]
#    
#    def __len__(self):
#        'Denotes the total number of samples'
#        return len(self.id_list)
#    
#    def __getitem__(self, index):
#        'Generates one sample of data'
#        # Select sample
#        ID = self.getID(index)
#        
#        with h5.File(self.hFile, 'r') as f :
#            # Load img and get label
#            img = self.loadH5Instance(f,"img/"+str(ID))
#            img = torch.from_numpy(np.transpose(img.astype(np.float32),(2,0,1)))
#            
#            label = self.loadH5Instance(f,"gt/"+str(ID))
#            label = torch.from_numpy(label).long()
#            
#            pixTL = self.loadH5Instance(f,"geo/"+str(ID))
#            gsd = self.loadH5Att(f,"geo","gsd")
#            epsg = self.loadH5Att(f,"geo","epsg")
#            
#        meta=torch.FloatTensor([ID,*pixTL,*gsd,epsg])
#        
#        #data augmentation
#        if self.transform :
#            #random crop
#            size1=self.cropSize[0]
#            size2=self.cropSize[1]
#            start1=np.random.randint(0,img.shape[1]-size1+1)
#            start2=np.random.randint(0,img.shape[2]-size2+1)
#            img=img[:,start1:start1+size1,start2:start2+size2]
#            label=label[start1:start1+size1,start2:start2+size2]
#            
#            #hflip and vflip
#            if np.random.random() < 0.5:
#                img=torch.flip(img, [2])
#                label=torch.flip(label, [1])
#            if np.random.random() < 0.5:
#                img=torch.flip(img, [1])
#                label=torch.flip(label, [0])
#            
#            #90° deg rotation
#            rot_nb=np.random.randint(0,4)
#            img=torch.rot90(img, rot_nb, [1, 2])
#            label=torch.rot90(label, rot_nb, [0, 1])
#        
##        tf.normalize(tensor, mean, std, inplace=False)
##        print(ID)
##        print(label.size())
##        print(img.size())
##        ost.plot2dArray(label.numpy())
##        ost.plot2dArray(torch.mean(img,0).numpy())
#        return img, label, meta