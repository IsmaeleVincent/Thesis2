# -*- coding: utf-8 -*-
"""
79-4U_77c88deg 
79-8U_76c76deg 
79-12U_75c64deg 
79-16U_74c52deg 
Created on Mon Mar 21 15:42:45 2022

@author: ismae
"""
"""
ATTENTION: All the blocks of this script completely re-write folders if they already exist
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
orig_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/"
fold_name = "NP829"
sorted_fold_path="/home/aaa/Desktop/Thesis2/Data from PSI/Sorted data/" #insert folder of sorted meausements files
renamed = sorted_fold_path+fold_name+"/Renamed/"
matrixes = sorted_fold_path+fold_name+"/Matrixes/"
pictures = sorted_fold_path+fold_name+"/Pictures/"
rawpictures = sorted_fold_path+fold_name+"/Raw pictures/"
th_matrixes = sorted_fold_path+fold_name+"/Theta matrixes/"
th_pictures = sorted_fold_path+fold_name+"/Theta pictures/"
th_rawpictures = sorted_fold_path+fold_name+"/Theta raw pictures/"
n_meas=208 #number of measurements files for each folder
n_pixel = 16384 #number of pixels in one measurement
n_runs = 8
n_theta = n_meas//n_runs

"""
This block renames the data files so they respect a specific labeling pattern: 
"Folder name_Number of measurement along theta", for example "D0043_001"
"""
# if os.path.exists(renamed):
#     shutil.rmtree(renamed)
# os.makedirs(renamed)
# for i in range (n_meas):
#     shutil.copy(orig_fold_path+fold_name +"/"+"D004"+str(int(3802+i))+".001", renamed+fold_name+"_"+str(i)+".mpa")

# """
# This block creates for each measurement a file with the matrix version of the 
# 128x128 measurements
# """
# if os.path.exists(matrixes):
#     shutil.rmtree(matrixes)
# os.makedirs(matrixes)
# for j in range (n_meas):
#     data = np.loadtxt(renamed+fold_name+"_"+str(j)+".mpa",skiprows=84, delimiter=",")
#     matrix = data.reshape(128, 128)
#     with open(matrixes+fold_name+"_"+str(j)+".mpa", 'w') as f:
#             np.savetxt(f, matrix, fmt="%1.0f")
# """
# This creates for each matrix file a corresponding picture
# """
# max_count = 0.
# matrixes1 = [np.loadtxt(matrixes+fold_name+"_"+str(j)+".mpa") for j in range (n_meas)]
# stack = np.stack(matrixes1,axis=2)
# max_count=np.amax(stack)
# if os.path.exists(pictures):
#     shutil.rmtree(pictures)
# os.makedirs(pictures)
# if os.path.exists(rawpictures):
#     shutil.rmtree(rawpictures)
# os.makedirs(rawpictures)
# for j in range (n_meas):
#     imageraw = im.fromarray(stack[:,:,j])
#     imageraw.save(rawpictures+fold_name+"_"+str(j)+'.tiff')
#     image = im.fromarray((stack[:,:,j]/max_count))
#     image.save(pictures+fold_name+"_"+str(j)+'.tiff')
   
"""
This crates the average counts for each theta 
"""
if os.path.exists(th_matrixes):
    shutil.rmtree(th_matrixes)
os.makedirs(th_matrixes)
matrixes1 = [np.loadtxt(matrixes+fold_name+"_"+str(j)+".mpa") for j in range(n_meas)]
stack = np.stack(matrixes1,axis=2)
for j in range (n_theta):
    matrix = np.sum(stack[:,:,j::n_theta],axis=2)/n_runs
    err=stack[:,:,j::n_theta].copy()
    for i in range(n_runs):
        err[:,:,i] = (err[:,:,i] - matrix)**2/(n_runs-1)
    err=np.sum(err,axis=2)**0.5
    with open(th_matrixes+fold_name+"_"+str(j)+".mpa", 'w') as f:
            np.savetxt(f, matrix, fmt="%1.0f")
    with open(th_matrixes+fold_name+"_"+str(j)+"_err.mpa", 'w') as f:
            np.savetxt(f, err, fmt="%1.0f")
max_count = 0.
matrixes = [np.loadtxt(th_matrixes+fold_name+"_"+str(j)+".mpa") for j in range (n_theta)]
stack = np.stack(matrixes,axis=2)
max_count=np.amax(stack)
if os.path.exists(th_pictures):
    shutil.rmtree(th_pictures)
os.makedirs(th_pictures)
if os.path.exists(th_rawpictures):
    shutil.rmtree(th_rawpictures)
os.makedirs(th_rawpictures)
for j in range (n_theta):
    imageraw = im.fromarray(stack[:,:,j])
    imageraw.save(th_rawpictures+fold_name+"_"+str(j)+'.tiff')
    image = im.fromarray((stack[:,:,j]/max_count))
    image.save(th_pictures+fold_name+"_"+str(j)+'.tiff')
