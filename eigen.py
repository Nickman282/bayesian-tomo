#Standard imports + DICOM
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

mem_file = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap')
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap')

first_moment_path = np.memmap(filename = mem_file_2, dtype='float32', mode='r', shape=(512**2,))
second_moment_path = np.memmap(filename = mem_file, dtype='float32', mode='w+', shape=(512**2,512**2))

print(first_moment_path.max())
print(second_moment_path.max())