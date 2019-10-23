#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Thu Jul 18 16:25:44 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""

import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py as h5
import OSToolBox as ost

def readH5Class(h5arg):
    with h5.File(h5arg, 'r') as f :
        return f.attrs["names"]

def plotcm(args):
    cm_file=args.cm
    
    
    conf_arr= np.loadtxt(cm_file,"int",delimiter=" ")
    print(conf_arr)
    
    if args.class_names is False:
        classes=[str(i) for i in range(conf_arr.shape[0])]
    else:
            classes=args.class_names
    
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    
    norm_conf=np.array(norm_conf)
#    norm_diag=[[np.nan if x!=y else norm_conf[x][y] for x in range(norm_conf.shape[1])] for y in range(norm_conf.shape[0])]
#    print(norm_conf)
#    print(norm_diag)

    fig = plt.figure(figsize=(8.5,7.5))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_conf, cmap='RdYlBu_r', 
                    interpolation='nearest')
#    res = ax.imshow(norm_diag, cmap='RdYlGn', 
#                    interpolation='nearest')
    
    width, height = conf_arr.shape
    if args.num :
        for x in range(width):
            for y in range(height):
                ax.annotate("{:.2f}".format(norm_conf[x][y]*100), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=args.ft_size,rotation=0)
    if args.class_names is False:
        rot=0
    else:
        rot=35
    cb = fig.colorbar(res,fraction=0.046, pad=0.04)
    plt.xticks(range(width), classes[:width],rotation=rot)
    cb.ax.tick_params(labelsize=args.ft_size) 
    plt.setp(ax.get_xticklabels(), fontsize=args.ft_size)
    plt.setp(ax.get_yticklabels(), fontsize=args.ft_size)
    plt.yticks(range(height), classes[:height])
    plt.savefig(ost.pathBranch(cm_file)+'/confusion_matrix.png', format='png')
    plt.savefig(ost.pathBranch(cm_file)+'/confusion_matrix.eps', format='eps')

if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-cm", type=str, help="path to cm file", required=True)
    parser.add_argument("-ft_size", type=int, help="font_size", default=10, required=False)
    parser.add_argument("-class_names", type=str, nargs='+', help="class names", default=False, required=False)
    parser.add_argument("-num", type=ost.str2bool, help="if number in mat", default=False, required=False)
    args = parser.parse_args()
    
    plotcm(args)