#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Mon Jul  1 11:00:36 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""

import OSToolBox as ost
from tqdm import tqdm
import argparse
import subprocess
import os


def main(args):
    #function to rebuild full image as TIF or VRT recursively from all small tiles
    #TIF version is almost deprecated 
    #VRT is way quicker
    
    #get folders and sort them by depth if recursive stitching
    if args.rec :
        l=ost.sortFoldersByDepth(*ost.checkFoldersWithDepth(args.dir))
    else :
        l=[args.dir]
    #display founded folders
    print([el[60:] for el in l])
    
    #recursive stitching
    for path in tqdm(l):
        current=os.getcwd()#save current console position
        os.chdir(ost.pathBranch(path))#move current console position to outAP to save instruction lenght
        
        #stitch current folder tiles 
        #"ulimit -s 65536;" is to avoid command line lenght restrictions"
        if args.vrt:
            out=path+"_merged.vrt"
            subprocess.call("ulimit -s 65536; gdalbuildvrt -vrtnodata "+ args.nodata +" "+out+" "+path+"/*.tif",shell=True)
        else:
            out=path+"_merged.tif"
            subprocess.call("ulimit -s 65536; gdal_merge.py -o "+ out+ " -n "+args.nodata+" -a_nodata "+ args.nodata+" "+ path+"/*.tif",shell=True)
        
        os.chdir(current)#move current console position orignel one
    return 0

if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, help="dir where to stitch ", required=True)
    parser.add_argument("-rec", type=ost.str2bool, help="bool true if recursive (but not last), if False dir is where tif are", default=False, required=False)
    parser.add_argument("-vrt", type=ost.str2bool, help="if true builds vrt instead of merging (faster calculs & less heavy but slower shows on qgis)", default=False, required=False)
    parser.add_argument("-nodata", type=str, help="no data value", default="99", required=False)
    args = parser.parse_args()
    
    #main
    main(args)