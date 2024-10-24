#Standard imports + DICOM
import scipy
import numpy as np
import pandas as pd
import os
import pydicom 
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

# CIL framework
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.framework import BlockDataContainer

from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator, \
                                       IdentityOperator, \
                                       GradientOperator, FiniteDifferenceOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, \
                                       L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, \
                                       TotalVariation, \
                                       ZeroFunction

# CIL Processors
from cil.processors import CentreOfRotationCorrector, Slicer

from cil.utilities.display import show2D

# Plugins
from cil.plugins.astra.processors import FBP, AstraBackProjector3D
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins import TomoPhantom

import imageio 

import torch

mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap')
mem_file_4 = Path('D:/Studies/MEng_Project/LIDC-IDRI/eigvec.pt')

covariance_path = np.memmap(filename = mem_file_3, dtype='float32', mode='r', shape=(256**2,256**2))

print(covariance_path.min())


#torch.cuda.empty_cache()
#print(torch.cuda.is_available())

cov_tensor = torch.from_numpy(covariance_path)

N = covariance_path.shape[0]

eigenvals, eigenvec = torch.lobpcg(cov_tensor, k=25)

print(eigenvals, eigenvec)

eigenvec = eigenvec.numpy()
gif_list = []
for i in range(eigenvec.T.shape[0]):
    gif_list.append(eigenvec.T[i].reshape(256, 256))

gif_list = np.array(gif_list)

#sort = np.flip(np.argsort(eigvals))[:25]

#eigvals, eigvec = eigvals[sort], eigenvec[sort]

imageio.mimsave('D:/Studies/MEng_Project/test_file35.gif', 1e9*gif_list, duration=1000)



fig, ax = plt.subplots()

ax.scatter(range(eigenvals.shape[0]), eigenvals)

plt.show()


#print(np.flip(np.sort(eigvals)).max())
