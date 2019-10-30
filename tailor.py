#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Tue Jul  2 14:46:23 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""
import os
import argparse
import sys
import OSToolBox as ost

import h5py as h5
from osgeo import gdal, ogr

import numpy as np
from tqdm import tqdm

import subprocess
import multiprocessing 
import shutil



def gtBuilder_v4(base_img, path, path_img, nodata_value, nodata_mask=False, split_field='', datatype=gdal.GDT_UInt16, arrayBack=False,colors=None):
    #Create Ground truth image given an base image extent by rasterizing a set of shapefiles from path
    #Return the class names or if arrayBack is True, return GT as array and class names
    #WARNING : order of read of shapefiles matters (natural order listing used here)
    #base_img : img on which groud truth extent is based
    #path : path of shape file or of folder where shapefiles are stored
    #path_img : path where to save ground truth image
    #split_field : deprecated
    #arrayBack: if true return GT as array and class names
    #nodata_mask : path to mask where GT will be produce. Values outside polygones will be set to no_data_values
    #color : color list in "r,g,b" may be deprecated
    path=os.path.abspath(path)
    print('image saved at: ', path_img)
    
    #manage case where path = shapefile or path = folder of shapefile
    list_shp=[]
    if os.path.isdir(path):
        list_shp=ost.getFileByExt(path,".shp")
    elif os.path.isfile(path):  
        list_shp.append(path)
    else:
        print('PATH GIVEN IS NOT SHAPE FILE OR DIR')
        return 0
    if len(list_shp)==0:
        print('NO SHAPE FILE FOUNDED')
        return 0
    
    #create GT raster by copying base img metadata
    base_img=os.path.abspath(base_img)
    base = gdal.Open(base_img)
    ras_c = ost.rasterCopy(base, path_img, datatype=datatype, bands=1)
    ras_c.GetRasterBand(1).SetNoDataValue(nodata_value)
    
    #get class names from shapefiles names
    names = [ost.pathLeaf(p) for p in list_shp]
    print(names)
    
    #if only 1 class, switch to binary mode and create other class and initialize GT with "others" class
    if len(names)==1:
        print("/!\ SWITCH TO CLASSIFICATION BINAIRE...")
        ras_c.GetRasterBand(1).Fill(1)
        names.append("others")
    
    #rasterize all shapefile in natural name order
    for i, path_shp in enumerate(list_shp):
        try:
            shp = ogr.Open(path_shp)
            if shp: # checks to see if shapefile was successfully defined
                print('loading: %s'%(path_shp))
            else: # if it's not successfully defined
                print( 'COULD NOT LOAD SHAPE: %s'%(path_shp))
        except: # Seems redundant, but if an exception is raised in the Open() call, you get a message
            print( 'EXCEPTION RAISED WHILE LOADING: %s'%(path_shp))# if you want to see the full stacktrace - like you are currently getting,# then you can add the following:
            raise
            
        source_layer=shp.GetLayer()
        gdal.RasterizeLayer(ras_c, [1], source_layer,burn_values=[i])#+1
        
    #apply color given
    if colors is not None :
        cT = gdal.ColorTable()
        # set color for each 
        for i in range(len(colors)):
            cT.SetColorEntry(i, colors[i])
        
        cT.SetColorEntry(nodata_value, (0,0,0,0))#nodata
        # set color table and color interpretation
        ras_c.GetRasterBand(1).SetRasterColorTable(cT)
        ras_c.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    
    if arrayBack : 
        gt=ras_c.ReadAsArray()
    
    #save raster to have it available for 
    ras_c=None
    
    #burn nodata zone with nodata_mask
    if nodata_mask is not False:
        print("Applying Nodata_Mask:",nodata_mask)
        gdal_instruc=["gdal_rasterize","-b","1","-i","-burn",str(nodata_value),nodata_mask,path_img]
        subprocess.call(gdal_instruc)#,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    
    if arrayBack :
        return gt, names
    else :
        return names



def multiprocessing_func(mlparg):
    #function to create an individual couple of image and GT tiles
    #launched by pool so can only take 1 arg which is a list of the arg you need.
    #extract args 
    discard=mlparg[0].discard
    marge=mlparg[0].margin
    nodata_value=mlparg[0].nodata_value
    verif_nodata=mlparg[0].verif_nodata
    
    list_border_norm=mlparg[2]
    ulx,uly,tsx,tsy, imgAP, gtAP,img_tile, gt_tile=mlparg[3]
    
    #case of deletion of empty tiles
    discarded=False
    
    #create img tile
    if img_tile is not None :
        srcwin=[ulx-marge,uly-marge, tsx+2*marge, tsy+2*marge]
        opts=gdal.TranslateOptions(format="GTIFF",outputType=gdal.GDT_Float32,srcWin=srcwin)
        out_ras=gdal.Translate(img_tile,imgAP,options=opts)
        
        #verif if not img = 0 on all pix of all band
        if discard:
            tot=0
            for i in range(1,out_ras.RasterCount+1):
                #fast way : calculate hist with large bucket so all pix fall in but not nodata pix so if only no data hist = 0
                # WARNING : IMG SHOULD HAVE A NODATA VALUE SET
                # this generate .aux.xml files
                hist=out_ras.GetRasterBand(i).GetHistogram(min=1, max=65536, buckets=1, approx_ok=False)
                tot+=np.sum(hist)
            if tot==0:
                discarded=True
#            print(tot,file)
        
        #discard or normalize and produce tf 
        if discarded:
            out_ras=None# Write to disk.
            os.remove(img_tile) #delete
        else:
            #TF is a mask array to know where to create GT
            tf=np.full((out_ras.GetRasterBand(1).YSize,out_ras.GetRasterBand(1).XSize),0)
            for i in range(1,out_ras.RasterCount+1):
                a=out_ras.GetRasterBand(i).ReadAsArray()
                tf+=np.array(a, dtype=bool) #store as bool aray where img is at 0 on each channel
#                print(tf)
                out_ras.GetRasterBand(i).WriteArray(
                  ost.normalize(a,force=True,mini=list_border_norm[i-1][0],maxi=list_border_norm[i-1][1] ))
            out_ras.FlushCache()  # Write to disk.
    
    
    #if we have discarded the img, don't create gt
    if not discarded and gt_tile is not None :
        srcwin=[ulx,uly, tsx, tsy]
        #if margin higher than 0, don't put GT on margin to avoid DATA LEAKING beetween train/validation/test sets
        if marge!=0:
            file_temp="/vsimem/"+ost.pathBranch(gt_tile)+"/"+ost.pathLeaf(gt_tile)+"temp.tif"
            opts=gdal.TranslateOptions(format="GTiff",outputType=gdal.GDT_Byte,srcWin=srcwin,noData=nodata_value)
            out_ds=gdal.Translate(file_temp,gtAP,options=opts)
            
            
            srcwin=[-marge,-marge, tsx+2*marge, tsy+2*marge]
            opts=gdal.TranslateOptions(format="GTiff",outputType=gdal.GDT_Byte,srcWin=srcwin,noData=nodata_value)
            out_ds=gdal.Translate(gt_tile,out_ds,options=opts)
        else:
            opts=gdal.TranslateOptions(format="GTiff",outputType=gdal.GDT_Byte,srcWin=srcwin,noData=nodata_value)
            out_ds=gdal.Translate(gt_tile,gtAP,options=opts)
        
        #final check to suppress GT on pix that have 0 value on each band thanks to TF array
        if verif_nodata:
            a=np.where(np.array(tf, dtype=bool),out_ds.GetRasterBand(1).ReadAsArray(),nodata_value)
            out_ds.GetRasterBand(1).WriteArray(a)
        
        out_ds.FlushCache()
    
    if discarded:
       return ost.pathLeafExt(img_tile),ost.pathLeafExt(gt_tile)
    else:
        return False



def megaTailor(args):
    #main function for args look at parser
    
    #check if gt is given as shp or tif. IF TIF NEED NOMENCLATRUE
    if args.create_gt:
        if ost.pathExt(args.gt)==".tif":
            gt_is_shp=False
            print("GT given as tif img")
            if args.nomenclature is False :
                print("ERROR : TIF GT NEED NOMENCLATURE")
                sys.exit()
        else :
            gt_is_shp=True
            print("GT given as shp")
    
    #create path variable
    imgAP=os.path.abspath(args.img)
    outAP=os.path.abspath(args.out)
    meta_img=outAP+"/meta_img.h5"
    meta_gt=outAP+"/meta_gt.h5"
    
    #check at least one of gt or tile are activated
    if not args.create_tile and not args.create_gt:
        print("ERROR : WAKE UP !! ... Choose at least one between GT or TILE !!")
        sys.exit()
    
    #extract main img vital information
    ds = gdal.Open(imgAP)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    tile_size_x = args.x
    tile_size_y = args.y
    lenght_dataset=(xsize//tile_size_x+1)*(ysize//tile_size_y+1)
    
    gtAP=None #initialize gtAP path variable
    
    #work GT
    if args.create_gt:
        gtAP=outAP+"/gt.tif"
        
        #Create GT img or crop it to imgAP extent
        if gt_is_shp:
            
            #get colors as dict for colortable
            if args.color is not False:
                print("Color given by :",args.color)
                c=np.loadtxt(args.color,delimiter=",",dtype=int)
                colors={}
                for i in range(c.shape[0]):
                    colors[int(c[i][0])]=(tuple(c[i][1:5]))
            else :
                colors =None
            
            #create gt and get class names from shapefiles names
            shpAP=os.path.abspath(args.gt)
            names = gtBuilder_v4(imgAP,shpAP,gtAP,nodata_value=args.nodata_value, nodata_mask=args.nodata_mask, datatype=gdal.GDT_Byte,colors=colors)
        
        else:
            geotransform = ds.GetGeoTransform()
            originX = geotransform[0]
            originY = geotransform[3]
            xres = geotransform[1]
            yres = geotransform[5]
            bottomX= originX+xsize*xres
            bottomY= originY+ysize*yres
            img_gtAP=os.path.abspath(args.gt)
            #crop GT TIF given to IMG extent
            gdal.Translate(gtAP,img_gtAP,projWin=[originX,originY,bottomX,bottomY],xRes=xres,yRes=yres,resampleAlg="nearest",noData=args.nodata_value)
            
        #Open newly created gt
        dsGT = gdal.Open(gtAP)
        band = dsGT.GetRasterBand(1)
        
        #replace names by nomenclature names given
        if args.nomenclature is not False:
            print("Class names given by :",args.nomenclature)
            names_indices=np.loadtxt(args.nomenclature,delimiter=" ",dtype=str)
#            indices=names_indices[:,0]
            names=list(names_indices[:,1])
#            print (names)
#            print (indices)
        
        # get class proportion in GT img
        hist=band.GetHistogram(approx_ok=False)
        prop = np.array(hist[0:len(names)])
        propPercent=100*prop/np.sum(prop)
        
        totalPix= dsGT.RasterXSize*dsGT.RasterYSize
        propPercentTotal=100*prop/totalPix
        str_rep='GT size = '+str(args.x)+'*'+str(args.y)+'*'+str(lenght_dataset)+' with '+str(int(np.sum(prop)))+'/'+str(totalPix)+' labeled pixels\n'+\
                "Dataset repartition :\n"+\
                '{:20} : {:12} || {:12} || {:4}'.format('CLASS','PIX NB', '% IN LABELED ','% IN TOTAL\n')+\
                '{:20} : {:12d} || {:12.2f}% || {:12.2f}%\n'.format('no_data',int(totalPix-np.sum(prop)), 0,(1-(np.sum(prop)/totalPix))*100)+\
                '\n'.join('{:20} : {:12d} || {:12.2f}% || {:12.2f}%'.format( \
                name, int(p), percent,pt) for name, percent,p,pt in zip(names,propPercent,prop,propPercentTotal))
        print(str_rep)
    
    # get img border 
    if args.create_tile:
        nb_channels=ds.RasterCount
        print("Getting img quartiles...")
        list_border_norm=[]
        
        #load band limit if given
        if args.band_norm is not False :
            list_border_norm=np.loadtxt(args.band_norm,dtype=float)
            #expand dim if only 1 band
            try :
                list_border_norm.shape[1]
            except :
                list_border_norm = np.expand_dims(list_border_norm, axis=0)
#                print(file_limit)
            #manage unmatching files
            if nb_channels!=list_border_norm.shape[0]:
                print("ERROR BAND_NORM FILE DOES NOT MATCH RASTER BAND COUNT :",nb_channels)
                for el in list_border_norm : print(el)
                sys.exit()
        else: #get border from img if not given (long to calculate for big image)
            for i in range(1,nb_channels+1):
                list_border_norm.append(ost.getBorderOutliers(ds.GetRasterBand(i).ReadAsArray(),lower=2, upper = 98))
       
        #display border found
        for i in range(1,nb_channels+1):
            print("Band {} normailzed to : {:7.2f}-{:7.2f}".format(
            i,list_border_norm[i-1][0],list_border_norm[i-1][1]))
    
    #create ou tiles folders and delete existant ones
    if args.create_tile:
        shutil.rmtree(outAP+"/tiles", ignore_errors=True)
        out_tiles = ost.createDir(outAP+"/tiles")
        output_filename = '/tile_'
        names_list_img=[]
    
    if args.create_gt :
        shutil.rmtree(outAP+"/tiles_gt", ignore_errors=True)
        out_gt = ost.createDir(outAP+"/tiles_gt")
        output_filename_gt = '/gt_'
        names_list_gt=[]
    
    
    #building tilling instructions
    instruc_list=[]
    count=0
    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            instructions=[None] * 5 #to have a stable form
            
            #img tiles instructions
            img_tile=None
            if args.create_tile:
                img_tile=out_tiles+output_filename+str(count)+"_"+str(i)+"_"+str(j)+".tif"
                instructions[2]=list_border_norm
            #gt tiles instructions
            gt_tile=None
            if args.create_gt:
                gt_tile=out_gt+output_filename_gt+str(count)+"_"+str(i)+"_"+str(j)+".tif"
            #general instructions
            instructions[0]=args
            instructions[1]=count
            instructions[3]=[i,j,tile_size_x,tile_size_y,imgAP, gtAP,img_tile, gt_tile]
            
            #store tiles names
            if args.create_tile :
                names_list_img.append(ost.pathLeafExt(img_tile))
            if args.create_gt :
                names_list_gt.append(ost.pathLeafExt(gt_tile))
            
            #append instructions
            instruc_list.append(instructions)
            count+=1
            
    #generating tiles with instruction
    print("Generating tiles...")
    list_discarded=[]
    with multiprocessing.Pool() as pool:
        for dis in tqdm(pool.imap_unordered(multiprocessing_func, instruc_list),total=len(instruc_list)):
            if dis is not False:
                list_discarded.append(dis)
    
    #register discarded tiles
    if args.discard is True :
       dis_file=outAP+"/discarded_tiles.txt"
       np.savetxt(dis_file,list_discarded,fmt='%s')
   #delete GT tiles if no img tiles and discard_file is given
    if args.discard_file is not False and args.create_tile is False:
        print("Suppressing tiles according to discard file :",args.discard_file,)
        list_discarded=np.genfromtxt(args.discard_file,dtype='str')
        for (ti,tg) in list_discarded :
            try:os.remove(out_gt+"/"+tg) #delete
            except:pass
    
    #rename tiles if discard operation to be sure to have all tiles with number from 0 to n without missing ones
    #renaming is vital for selecter algorithm
    if args.discard or args.discard_file is not False:
        print("Renaming tiles...")
        if args.create_tile :
            to_rename_img=ost.getFileByExt(out_tiles,".tif")
            root=ost.pathBranch(ost.pathBranch(to_rename_img[0]))
            for i,name in tqdm(enumerate(to_rename_img)):
                old_name=ost.pathLeaf(name)
                name_split=old_name.split("_")
                x_name=name_split[2]
                y_name=name_split[3]
                new_name_img=root+"/tiles/tile_"+str(i)+"_"+str(x_name)+"_"+str(y_name)+".tif"
                os.rename(name,new_name_img)
            names_list_img=ost.getFileByExt(out_tiles,".tif")
        
        if args.create_gt :
            to_rename_gt=ost.getFileByExt(out_gt,".tif")
            root=ost.pathBranch(ost.pathBranch(to_rename_gt[0]))
            for i,name in tqdm(enumerate(to_rename_gt)):
                old_name=ost.pathLeaf(name)
                name_split=old_name.split("_")
                x_name=name_split[2]
                y_name=name_split[3]
                new_name_gt=root+"/tiles_gt/gt_"+str(i)+"_"+str(x_name)+"_"+str(y_name)+".tif"
                os.rename(name,new_name_gt)
            names_list_gt=ost.getFileByExt(out_gt,".tif")
    
    #remove all stats files generated for mono bucket analysis in multiprocessing_func
    if args.create_tile :
        [os.remove(a) for a in ost.getFileByExt(out_tiles,".aux.xml")]
        
    
    #writting h5 file of metadata and delete already existing ones
    print("Writting h5 file...",end='')
    if args.create_tile:
        try:
            os.remove(meta_img)
        except:
            pass
        with h5.File(meta_img, "w") as f:
            f.attrs["base_img"]=imgAP
            f.attrs["len"]=len(names_list_img)
            
            f.attrs["margin"]=args.margin
            f.attrs["channels"]=nb_channels
            f.attrs["norm_border"]=list_border_norm
            a=[ost.pathLeafExt(a) for a in names_list_img]
            f.create_dataset("tile_path",data=np.array(a,dtype=h5.special_dtype(vlen=str)))
    if args.create_gt:
        try:
            os.remove(meta_gt)
        except:
            pass
        with h5.File(meta_gt, "w") as f:
            f.attrs["base_img"]=imgAP
            f.attrs["len"]=len(names_list_gt)
            
            f.attrs["info"]=str_rep
            f.attrs["names"]=names
            f.attrs["class_weight"]=propPercent/100
            a=[ost.pathLeafExt(a) for a in names_list_gt]
            f.create_dataset("gt_path",data=np.array(a,dtype=h5.special_dtype(vlen=str)))
    print(" done.")
    
    
    #creating data selection folder and shapefiles
    if args.create_select:
        current=os.getcwd()#save current console position
        os.chdir(outAP)#move current console position to outAP to save instruction lenght
        print("Generating index shape of tiles...",end='')
        shutil.rmtree(outAP+"/data_selection", ignore_errors=True)
        out_selec = ost.createDir(outAP+"/data_selection")
        out_shp_index=ost.pathLeaf(out_selec)+"/all_index.shp"
        try :
            num=ost.getFileByExt(out_tiles,".tif")
        except:
            num=ost.getFileByExt(out_gt,".tif")
        gdalTIndex_instruc=["gdaltindex","-f","ESRI Shapefile","-t_srs","EPSG:2154",out_shp_index,*num]
        subprocess.call(gdalTIndex_instruc,stdout=subprocess.DEVNULL)
        os.chdir(current)#move current console position orignel one
        
        #copy all into to create others index
        list_TIndex = ost.getFileBySubstr(out_selec,"all_index")
        index_names=["test_index","val_index","train_index"]
        for n in index_names:
            for a in list_TIndex :
                shutil.copy(a, out_selec+"/"+n+ost.pathExt(a)) 
        print(" done.")
    
    return 0


if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    #essentials
    parser.add_argument("-gt", type=str, help="shape: input .shp or shp-dir OR TIF img with classif", required=True)
    parser.add_argument("-img", type=str, help="tif image: img to apply shp for GT", required=True)
    parser.add_argument("-out", type=str, help="path: GT img saving path", default='', required=False)
    #values
    parser.add_argument("-x", type=int, help="int: dim of tile in pixels", default=1100, required=False)
    parser.add_argument("-y", type=int, help="int: dim of tile in pixels", default=1000, required=False)
    parser.add_argument("-nodata_value", type=int, help="no data value in images", default=99, required=False)
    parser.add_argument("-margin", type=int, help="marging to take for images", default=0, required=False)
    #boolean
    parser.add_argument("-create_tile", type=ost.str2bool, help="create img tiles from img",default=True, required=False)
    parser.add_argument("-create_gt", type=ost.str2bool, help="create GT tiles from gt",default=True, required=False)
    parser.add_argument("-create_select", type=ost.str2bool, help="if true, create shapes for data selection", default=True, required=False)
    parser.add_argument("-discard", type=ost.str2bool, help="if true, discard all empty tiles from set",default=False, required=False)
    parser.add_argument("-verif_nodata", type=ost.str2bool, help="if true, makes sure you do not have 0-on-all-channels  annotated pixels /!\ may disturb your image",default=False, required=False)
    
    #path or false
    parser.add_argument("-band_norm", type=ost.SFParser, help="path to band normalization values file", default=False, required=False)
    parser.add_argument("-nomenclature", type=ost.SFParser, help="link to nomenclature if img", default=False, required=False)
    parser.add_argument("-color", type=ost.SFParser, help="path to color txt file as value,r,g,b,a per line for each value to color", default=False, required=False)
    parser.add_argument("-nodata_mask", type=ost.SFParser, help="path to nodata mask", default=False, required=False)
    parser.add_argument("-discard_file", type=ost.SFParser, help="path to discarded tiles", default=False, required=False)

    args = parser.parse_args()
    megaTailor(args)
