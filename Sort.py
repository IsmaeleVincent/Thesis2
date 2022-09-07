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
from PIL import Image as im

sorted_fold_path="/home/aaa/Desktop/Thesis2/Divergence/Sorted data/nd061716/" #insert folder of sorted meausements files
allmeasurements = sorted_fold_path+"All measurements/"
allrenamed = allmeasurements +"All renamed/"
allmatrixes = allmeasurements + "All matrixes/"
allpictures = allmeasurements + "All pictures/"
allrawpictures = allmeasurements + "All raw pictures/"
foldername=["Off-Bragg"]
n_theta=[40] #number of measurements files for each folder
n_pixel = 16384 #number of pixels in one measurement
"""
This block renames the data files so they respect a specific labeling pattern: 
"Tilt angle_number of measurement along theta", for example "0deg_001"
It has to be done folder by folder
"""
orig_fold_path="/home/aaa/Desktop/Thesis2/Divergence/nd061716/"
fold_name = "Off-Bragg"
renamed = sorted_fold_path +fold_name+"/Renamed/"
if os.path.exists(renamed):
    shutil.rmtree(renamed)
os.makedirs(renamed)
for i in [0,70]:
    shutil.copy(orig_fold_path+fold_name +"/nd061716_zeta"+str("%1d" % (i,))+".mpa", renamed+fold_name+"_zeta"+str("%1d" % (i,))+".mpa")

"""
This block creates for each measurement a file with the matrix version of the 
128x128 measurements
"""
data=np.zeros(n_pixel);
for k in [0]:#range(len(foldername)):
    matrixes = sorted_fold_path+foldername[k]+"/Matrixes/"
    if os.path.exists(matrixes):
        shutil.rmtree(matrixes)
    os.makedirs(matrixes)
    for j in [0,70]:
        data=np.loadtxt(sorted_fold_path +foldername[k]+"/Renamed/"+foldername[k]+"_zeta"+str("%1d" % (j,))+".mpa",skiprows=125, max_rows=n_pixel)[::-1]
        data_matrix = data.reshape(128, 128)
        with open(matrixes+foldername[k]+"_zeta"+str("%1d" % (j,))+'.mpa', 'w') as f:
                np.savetxt(f, data_matrix, fmt="%1.0f")
"""
This creates for each matrix file a corresponding picture
"""
max_count = 0.
for k in [0]:
    for j in [0,70]:
        f = sorted_fold_path+foldername[k]+"/Matrixes/"+foldername[k]+"_zeta"+str("%1d" % (j,))+".mpa"
        data_matrix = np.loadtxt(f)
        if np.amax(data_matrix)>max_count:
            max_count=np.amax(data_matrix)
for k in range(len(foldername)):
    pictures = sorted_fold_path+foldername[k]+"/Pictures/"
    if os.path.exists(pictures):
        shutil.rmtree(pictures)
    os.makedirs(pictures)
    rawpictures = sorted_fold_path+foldername[k]+"/Raw pictures/"
    if os.path.exists(rawpictures):
        shutil.rmtree(rawpictures)
    os.makedirs(rawpictures)
    for j in [0,70]:
        f = sorted_fold_path+foldername[k]+"/Matrixes/"+foldername[k]+"_zeta"+str("%1d" % (j,))+".mpa"
        data_matrix = np.loadtxt(f)/max_count
        data_raw = np.loadtxt(f)
        image = im.fromarray((data_matrix))
        image.save(pictures+foldername[k]+"_zeta"+str("%1d" % (j,))+'.tiff')
        imageraw = im.fromarray(data_raw)
        imageraw.save(rawpictures+foldername[k]+"_zeta"+str("%1d" % (j,))+'.tiff')

"""
This creates common folders "Renamed", "Matrixes" and "Pictures" for all measurements at all tilt angles 
"""

# # if os.path.exists(allmeasurements):
# #         shutil.rmtree(allmeasurements)
# # os.makedirs(allmeasurements)
# if os.path.exists(allpictures):
#         shutil.rmtree(allpictures)
# os.makedirs(allpictures)
# if os.path.exists(allrawpictures):
#         shutil.rmtree(allrawpictures)
# os.makedirs(allrawpictures)
# if os.path.exists(allmatrixes):
#     shutil.rmtree(allmatrixes)
# os.makedirs(allmatrixes)
# if os.path.exists(allrenamed):
#     shutil.rmtree(allrenamed)
# os.makedirs(allrenamed)

# for k in range(len(foldername)):
#     for j in range (1,n_theta[k]+1):
#         filename = foldername[k]+"_"+str("%03d" % (j,))+".mpa"
#         picname = foldername[k]+"_"+str("%03d" % (j,))+".tiff"
#         folder = sorted_fold_path+foldername[k]
#         shutil.copy(folder+"/Renamed/"+filename, allrenamed+filename)
#         shutil.copy(folder+"/Matrixes/"+filename, allmatrixes+filename)
#         shutil.copy(folder+"/Pictures/"+picname, allpictures+picname)
#         shutil.copy(folder+"/Raw pictures/"+picname, allrawpictures+picname)
