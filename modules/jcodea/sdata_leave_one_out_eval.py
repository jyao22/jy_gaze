#!/usr/bin/env python
# coding: utf-8

import os
import argparse


# aparse.ArgumentParser(
#         description='gaze estimation using binned loss function.')
#     parser.add_argument(
#         '--evalpath', dest='evalpath', help='path for evaluating gaze test.',
#         default="evaluation\L2CS-gaze360-_standard-10", type=str)
#     parser.add_argument(
#         '--respath', dest='respath', help='path for saving result.',
#         default="evaluation\L2CS-gaze360-_standard-10", type=str)


# args = parse_args()
# evalpath =args.evalpath
# respath=args.respath


evalpath = '../results/soutput/evaluation'
respath = '../results/soutput/eval'
n_epochs = 60


if not os.path.exists(respath):
        os.makedirs(respath)
        


print(evalpath)





dirlist = os.listdir(evalpath)
dirlist.sort()
print(dirlist)





with open(os.path.join(respath,"avg.log"), 'w') as outfile:    
    outfile.write("Average equal\n")
    min=10.0
    dirlist = os.listdir(evalpath)
    dirlist.sort()
    n_folds = len(dirlist)
    l = 0.0
    for j in range(n_epochs):  # j = epoch-1
#         j=20   # strange?
        avg = 0.0
        h = j+3  # remove 3 lines at the top
        for i in dirlist:  # traverse each fold
            with open(evalpath + "/" + i + "/mpiigaze.log") as myfile: #resad the save file for jth epoch

                x=list(myfile)[h]
#                 print(i,j+1, x)
                str1 = "" 

                # traverse in the string  
                for ele in x: 
                    str1 += ele 
#                 print("x :", x)
#                 print("str1:",str1)
                split_string = str1.split("MAE:",1)[1]  # 1 is the max number of split/separation
                avg+=float(split_string)
        avg = avg/n_folds  #average MAE for the specified epoch
        if avg < min:
            min = avg
            l = j + 1
        outfile.write("epoch"+str(j+1)+"= "+str(avg)+"\n")
    outfile.write("min angular error equal= "+str(min)+"at epoch= "+str(l)+"\n")
print("min angular error equal= "+str(min)+"at epoch= "+str(l)+"\n")   
print(min)

