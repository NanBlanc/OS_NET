#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 
    Wed May 29 10:06:20 2019
Author: 
    Olivier Stocker 
    NanBlanc
Project: 
    OLIVENET    
"""
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import OSToolBox as ost



class SavePerf:
#this class is not modular => NEED IMPROVEMENT Work only for 3 datasat train/val/test
    def __init__(self, epoch,outDir, classes, dataset_names=["Train","Val","Test"]):
    #initialize all containers
        self.epoch=epoch
        self.outDir=outDir
        self.classes=classes
        self.dataset_names=dataset_names
        
        #general container
        self.list_cm=[]
        self.list_loss=[]
        self.list_mIoU=[]
        self.list_ious=[]
        self.list_oa=[]
        self.list_register=[]#container for each metric epoch (thus depend of args.mem of MAIN algorithm)
        self.list_loss_register=[]##container for each loss epoch (thus all epochs)
        
        #register best epochs in the two register container
        self.best=0
        self.best_id=0
        
        #add one container for each dataset in each general container
        for i in range(len(dataset_names)) :
            self.list_cm.append([])
            self.list_loss.append([])
            self.list_mIoU.append([])
            self.list_ious.append([])
            self.list_oa.append([])
            self.list_register.append([])
            self.list_loss_register.append([])
    
    def addLossResults(self,i_epoch,loss,dataset_instance,show=1):
        #function to save loss and print it
        self.list_loss[dataset_instance].append(loss)
        self.list_loss_register[dataset_instance].append(i_epoch)
        if show:
            #color given the dataset instance
            if dataset_instance==0: color = (19,161,14
            elif dataset_instance==1: color = (193,156,0)
            else: color = (204,0,0)
            print(ost.PRINTCOLOR(*color),end="")#set color
            text = self.dataset_names[dataset_instance]
            print('Epoch %3d -> %s Loss: %1.6f' % (i_epoch, text, loss))
            print(ost.RESETCOLOR,end="")#reset color
    
    def addFullResults(self,i_epoch,cm,loss,dataset_instance,show=1):
        #function to save loss and other metric from CM
        #RETURN : IS_BEST only if dataset instance is 1 (VALIDATION set) 
        #ISBEST is TRUE if current epoch got best mIoU ever
        
        #color given the dataset instance
        if dataset_instance==0: color = (19,161,14)  
        elif dataset_instance==1: color = (193,156,0) 
        else: color = (204,0,0)
        #get dataset instance name (train/val/test)
        text = self.dataset_names[dataset_instance]
        
        print(ost.PRINTCOLOR(*color),end="")#set color
        mIoU, ious= cm.class_IoU(show)#calcul iou and miou and display IoU
        oa=cm.overall_accuracy()#calcul OA
        #display general metrics
        if show:
            print('Epoch %3d -> %s Overall Accuracy: %3.2f%% %s mIoU : %3.2f%% %s Loss: %1.6f' \
                  % (i_epoch, text, oa,text, mIoU,text, loss))
        print(ost.RESETCOLOR,end="")#reset color
        
        #save metric values in containers
        self.list_cm[dataset_instance].append(cm)
        self.list_loss[dataset_instance].append(loss)
        self.list_mIoU[dataset_instance].append(mIoU)
        self.list_ious[dataset_instance].append(ious)
        self.list_oa[dataset_instance].append(oa)
        self.list_register[dataset_instance].append(i_epoch)
        self.list_loss_register[dataset_instance].append(i_epoch)
        
        #check if current epoch is best epoch by comparing mIoU
        isBest = True if mIoU>=max(self.list_mIoU[dataset_instance]) else False
        if isBest and dataset_instance==1: #verif if we really are in VALIDATION DATASET INSTANCE
            self.best=i_epoch
            self.best_id=self.list_mIoU[dataset_instance].index(mIoU)
        return isBest
    
    def printResultsTxt(self):
        #SAVE at given self.outDir all container values
        
        #loss
        d= {'epochs '+self.dataset_names[0]:self.list_loss_register[0],\
                         'loss '+self.dataset_names[0]:self.list_loss[0],\
                         'epochs '+self.dataset_names[1]:self.list_loss_register[1],\
                         'loss '+self.dataset_names[1]:self.list_loss[1],\
                         'epochs '+self.dataset_names[2]:self.list_loss_register[2],\
                         'loss '+self.dataset_names[2]:self.list_loss[2]}
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(self.outDir+"/loss.csv", sep=";", index=False)
        
        #OA
        d={'epochs '+self.dataset_names[0]:self.list_register[0],\
                         'OA '+self.dataset_names[0]:self.list_oa[0],\
                         'epochs '+self.dataset_names[1]:self.list_register[1],\
                         'OA '+self.dataset_names[1]:self.list_oa[1],\
                         'epochs '+self.dataset_names[2]:self.list_register[2],\
                         'OA '+self.dataset_names[2]:self.list_oa[2]}
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(self.outDir+"/oa.csv", sep=";", index=False)
        
        #IOU
        d={'epochs '+self.dataset_names[0]:self.list_register[0],\
                         'mIoU '+self.dataset_names[0]:self.list_mIoU[0],\
                         'epochs '+self.dataset_names[1]:self.list_register[1],\
                         'mIoU '+self.dataset_names[1]:self.list_mIoU[1],\
                         'epochs '+self.dataset_names[2]:self.list_register[2],\
                         'mIoU '+self.dataset_names[2]:self.list_mIoU[2]}
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(self.outDir+"/mIoU.csv", sep=";", index=False)
        
        #IOU per classe
        ar_iou=np.array(self.list_ious)
        for i in range(len(self.classes)):
            d={'epochs '+self.dataset_names[0]:self.list_register[0],\
                             'IoU '+self.classes[i]+self.dataset_names[0]:ar_iou[0][:,i]*100,\
                             'epochs '+self.dataset_names[1]:self.list_register[1],\
                             'IoU '+self.classes[i]+self.dataset_names[1]:ar_iou[1][:,i]*100,\
                             'epochs '+self.dataset_names[2]:self.list_register[2],\
                             'IoU '+self.classes[i]+self.dataset_names[2]:ar_iou[2][:,i]*100}
            df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
            df.to_csv(self.outDir+"/"+self.classes[i]+"_IuO.csv", sep=";", index=False)
        
        #best epoch
        text=['Loss','OA','mIoU']+["IoU "+cn for cn in self.classes]
        d={"":text}
        for i in range(len(self.dataset_names)):
            idx=self.best_id if i!=0 else self.best
            data=[self.list_loss[i][idx]]#LOSS FOR TRAIN IS EVERY EPOCH SO USE self.best BUT OTHER USES self.best_id
            data.append(self.list_oa[i][self.best_id])
            data.append(self.list_mIoU[i][self.best_id])
            for j in range(len(self.classes)):
                data.append(ar_iou[i][self.best_id,j]*100)
            d={self.dataset_names[i]:data}
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(self.outDir+"/bestPerf.csv", sep=";", index=False)
    
    def printResults(self): 
        #PLOT at given self.outDir all container values
        
        #LOSS
        fig,ax=plt.subplots()
        ax.plot(self.list_loss_register[0] ,self.list_loss[0],'g',label=self.dataset_names[0])
        ax.plot(self.list_loss_register[1] ,self.list_loss[1],'y',label=self.dataset_names[1])
        ax.plot(self.list_loss_register[2] ,self.list_loss[2],'r',label=self.dataset_names[2])
        ax.axvline(x=self.best,color='k')
        ax.annotate('Best epoch: %d\nTrain: %.3f\nVal: %.3f\nTest: %.3f'
                    #LOSS FOR TRAIN IS EVERY EPOCH SO USE self.best BUT OTHER USES self.best_id
                    %(self.best,self.list_loss[0][self.best],self.list_loss[1][self.best_id],self.list_loss[2][self.best_id]),
                    (0.025, 0.805), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9",alpha=0.2))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss according to epochs')
        ax.legend(loc='upper right')
        plt.grid(True)
        plt.ylim(top=2.5,bottom = 0)
        plt.savefig(self.outDir+"/loss.tif")
        
        #OA
        fig,ax=plt.subplots()   
        ax.plot(self.list_register[0] ,self.list_oa[0],'g',label=self.dataset_names[0])
        ax.plot(self.list_register[1] ,self.list_oa[1],'y',label=self.dataset_names[1])
        ax.plot(self.list_register[2] ,self.list_oa[2],'r',label=self.dataset_names[2])
        ax.axvline(x=self.best,color='k')
        ax.annotate('Best epoch: %d\nTrain: %.3f\nVal: %.3f\nTest: %.3f'
                    #LOSS FOR TRAIN IS EVERY EPOCH SO USE self.best BUT OTHER USES self.best_id
                    %(self.best,self.list_oa[0][self.best_id],self.list_oa[1][self.best_id],self.list_oa[2][self.best_id]),
                    (0.025, 0.805), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9",alpha=0.2))
        plt.xlabel('Epochs')
        plt.ylabel('Overall Accuracy (%)')
        plt.title('OA according to epochs')
        ax.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(self.outDir+"/oa.tif")
        
        #mIOU
        fig,ax=plt.subplots()   
        ax.plot(self.list_register[0] ,self.list_mIoU[0],'g',label=self.dataset_names[0])
        ax.plot(self.list_register[1] ,self.list_mIoU[1],'y',label=self.dataset_names[1])
        ax.plot(self.list_register[2] ,self.list_mIoU[2],'r',label=self.dataset_names[2])
        ax.axvline(x=self.best,color='k')
        ax.annotate('Best epoch: %d\nTrain: %.3f\nVal: %.3f\nTest: %.3f'
                    #LOSS FOR TRAIN IS EVERY EPOCH SO USE self.best BUT OTHER USES self.best_id
                    %(self.best,self.list_mIoU[0][self.best_id],self.list_mIoU[1][self.best_id],self.list_mIoU[2][self.best_id]),
                    (0.025, 0.805), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9",alpha=0.2))
        plt.xlabel('Epochs')
        plt.ylabel('mIoU (%)')
        plt.title('mIoU according to epochs')
        ax.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(self.outDir+"/mIoU.tif")
        
        #IOU per classe
        ar_iou=np.array(self.list_ious)
        for i in range(len(self.classes)):
                    #display result
            fig,ax=plt.subplots()  
            ax.plot(self.list_register[0],ar_iou[0][:,i]*100,'g',label=self.dataset_names[0])
            ax.plot(self.list_register[1],ar_iou[1][:,i]*100,'y',label=self.dataset_names[1])
            ax.plot(self.list_register[2],ar_iou[2][:,i]*100,'r',label=self.dataset_names[2])
            ax.axvline(x=self.best,color='k')
            ax.annotate('Best epoch: %d\nTrain: %.3f\nVal: %.3f\nTest: %.3f'
                    #LOSS FOR TRAIN IS EVERY EPOCH SO USE self.best BUT OTHER USES self.best_id
                    %(self.best,ar_iou[0][self.best_id,i]*100,ar_iou[1][self.best_id,i]*100,ar_iou[2][self.best_id,i]*100),
                    (0.025, 0.805), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="0.9",alpha=0.2))
            plt.xlabel('Epochs')
            plt.ylabel('IoU (%)')
            plt.title('IoU of '+self.classes[i] + " according to epochs")
            ax.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(self.outDir+"/"+self.classes[i]+"_IuO.tif")
            plt.clf()



class ConfusionMatrix:
    #class to manage confusion matrix and compute associated metrics
    def __init__(self, n_class, class_names,noDataValue):
        self.CM=np.zeros((n_class,n_class))
        self.n_class=n_class
        self.class_names=class_names
        self.noDataValue=noDataValue
        
    def clear(self):
        self.CM=np.zeros((self.n_class,self.n_class))
    
    def add_batch(self, gt, pred):
        labeled= gt!=self.noDataValue
        if labeled.any():
            self.CM+=confusion_matrix(gt[labeled], pred[labeled], labels = list(range(0,self.n_class)))
    
    def add_matrix(self,matrix):
        self.CM+=matrix
        
    def overall_accuracy(self):
        return 100*self.CM.trace()/self.CM.sum()
    
    def class_IoU(self, show=1):
        #RETURN : miou & list of ious
        #show = True for printing IoU
        ious = np.full(self.n_class, 0.)
        for i_class in range(self.n_class):
            diviseur=(self.CM[i_class,:].sum()+self.CM[:,i_class].sum()-self.CM[i_class,i_class])
            if diviseur ==0:
#                print("WAS ZERO")
                ious[i_class]=np.nan
            else:
#                print("WAS NOT ZERO")
                ious[i_class] = self.CM[i_class,i_class] /diviseur
        if show:
            print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
        return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum(), ious
    
    def printPerf(self,outDir):
        #calculate and output score
        np.savetxt(outDir+"/precision.txt",self.class_precision()*100,fmt="%.4f")
        np.savetxt(outDir+"/recall.txt",self.class_recall()*100,fmt="%.4f")
        np.savetxt(outDir+"/f1_score.txt",self.class_f1_score()*100,fmt="%.4f")
        np.savetxt(outDir+"/oa.txt",[self.overall_accuracy()],fmt="%.4f")
        #calculate & output IoU and mIoU
        mious=self.class_IoU()
        np.savetxt(outDir+"/miou.txt",[mious[0]],fmt="%.4f")
        np.savetxt(outDir+"/iou.txt",mious[1]*100,fmt="%.4f")
        #output CM
        np.savetxt(outDir+"/cm.txt",self.CM,fmt="%d")
        
    def class_precision(self):
        self.precision=[row[i]/sum(row) for i,row in enumerate(self.CM)]
        return self.precision
    
    def class_recall(self):
        self.recall=[row[i]/sum(row) for i,row in enumerate(self.CM.T)]
        return self.recall
    
    def class_f1_score(self):
        self.class_precision()
        self.class_recall()
        self.f1_score=[2*p*r/(p+r) for (p,r) in zip(self.precision,self.recall)]
        return self.f1_score


##### MAIN JUST TO CHECK CODING
if __name__ == '__main__':
    y_t=np.array([[0,1,2,1,2,5],[0,1,3,4,4,5],[4,2,3,0,5,1],[3,4,4,1,2,2],[1,1,5,1,2,3]])
    y_p=np.array([[1,1,2,1,2,4],[0,1,3,4,4,2],[4,2,3,0,4,1],[3,4,4,1,2,2],[1,1,1,1,2,3]])
    print(y_t)
    print(y_p)
    
    labeled = y_t!=5
    print(labeled)
    print(y_t[labeled])
    print(y_p[labeled])
    
    cm = ConfusionMatrix(5, ['bati', 'veget', 'eau', 'champs', 'route'],5)
    cm.add_batch(y_t[labeled],y_p[labeled])
    
    OA=cm.overall_accuracy()
    IoU = cm.class_IoU()
    print('Overall Accuracy: %3.2f%% mIoU : %3.2f%% ' % (OA, IoU[0]))
    print(cm.CM)
    print(cm.class_precision())
    print(cm.class_recall())
    print(cm.class_f1_score())
    
    #check with sklearn
    from sklearn.metrics import f1_score
    f1=f1_score(y_t[labeled], y_p[labeled], average=None)
    print(f1)
    
    y_t=np.ones((5,5))
    y_p=np.ones((5,5))
    print(y_t)
    print(y_p)
    
    labeled = y_t!=5
    print(labeled)
    print(y_t[labeled])
    print(y_p[labeled])
    cm2 = ConfusionMatrix(5, ['bati', 'veget', 'eau', 'champs', 'route'],5)
    cm2.add_batch(y_t[labeled],y_p[labeled])
    cm.add_matrix(cm2.CM)
    cm.printPerf("/media/ostocker/Data/OS/trash_tests/cm_perf")