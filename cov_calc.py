# Standard imports + DICOM
import numpy as np
import pandas as pd
import os
import pydicom 
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
from tqdm import tqdm

mem_file = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap')
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap')
mem_file_3 = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat_com.mymemmap')

first_moment_path = np.memmap(filename = mem_file_2, dtype='float32', mode='r', shape=(256**2,))
second_moment_path = np.memmap(filename = mem_file, dtype='float32', mode='r', shape=(256**2,256**2))
covariance_path = np.memmap(filename = mem_file_3, dtype='float32', mode='w+', shape=(256**2,256**2))

fig, ax = plt.subplots()
im = ax.imshow(np.exp(first_moment_path.reshape(256, 256)), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax)
plt.show()

#print(first_moment_path.max())
#print(second_moment_path.min())


print("Covariance Calculation:")
for sub_i in tqdm(range(255)):
    sec_st_i = 255*sub_i
    sec_en_i = 255*sub_i + 255
    for sub_j in range(255):
        sec_st_j = 255*sub_j
        sec_en_j = 255*sub_j + 255

        mul_exp = first_moment_path.reshape(-1, 1)[sec_st_i : sec_en_i, :] @ first_moment_path.reshape(1, -1)[:, sec_st_j : sec_en_j]
        covariance_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = second_moment_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] - mul_exp


print(covariance_path.min())
print(covariance_path.max())




