# Standard imports + DICOM
import numpy as np
import pandas as pd
import os
import pydicom 
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from PIL import Image

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

from tqdm import tqdm
import imageio

mem_file = Path('D:/Studies/MEng_Project/LIDC-IDRI/cov_mat.mymemmap')
mem_file_2 = Path('D:/Studies/MEng_Project/LIDC-IDRI/means.mymemmap')

first_moment_path = np.memmap(filename = mem_file_2, dtype='float32', mode='w+', shape=256**2)
second_moment_path = np.memmap(filename = mem_file, dtype='float32', mode='w+', shape=(256**2,256**2))


# Create a Dataset class to extract and hold relevant information from the dataset
class Dataset:
    
    # Initially only takes in main datapath and the standard chunk size for the data processing
    def __init__(self, main_path):

        self.main_path = main_path
        print("Initial rescale load-in:")
        self.rescaler_init()

    def chunk_loadin(self, chunk_size, chunk_ind=0, tag_list=None):
        self.chunk_size = chunk_size
        if tag_list == None:
            self.tag_list = [[0x0010, 0x0020]]
        else:
            self.tag_list = [[0x0010, 0x0020], *tag_list]

        try:
            main_dir_list = os.listdir(self.main_path)[chunk_ind*self.chunk_size:chunk_ind*self.chunk_size + self.chunk_size]
            self.actual_chunk_size = self.chunk_size
        except:
            try:
                main_dir_list = os.listdir(self.main_path)[chunk_ind*self.chunk_size:]
                self.actual_chunk_size = len(main_dir_list)
            except:
                ValueError

        prev_idx = None
        self.tag_names = []

        # Load data as list of dicts
        super_list = []
        print("Patients Load-In Progress:")
        for patient in tqdm(main_dir_list):
            patient_file_dirs = glob(f"{self.main_path}/{patient}/*/*/*.dcm")

            patient_list = []
            for slice_filepath in patient_file_dirs:
                slice_file = pydicom.dcmread(slice_filepath)              

                if slice_file[0x0008,0x0060].value == "CT": 

                    try:    
                        temp_dict = {}
                        for i in range(len(self.tag_list)):

                            temp_name = slice_file[self.tag_list[i][0],self.tag_list[i][1]].name

                            if prev_idx != i:
                                self.tag_names.append(temp_name)
                                prev_idx = i

                            if self.tag_list[i] == [0x7FE0, 0x0010]: 
                                temp_dict[f"{temp_name}"] = slice_file.pixel_array

                            else:
                                temp_dict[f"{temp_name}"] = slice_file[self.tag_list[i][0],self.tag_list[i][1]].value

                        super_list.append(temp_dict)

                    except:
                        continue   

                else:
                    continue
                
        # LD to DL
        self.tag_dict_main = {}
        print("Conversion to Dataframe:")
        for tag_name in tqdm(self.tag_names):
            super_tag_list = []
            for slice_t in super_list:
                super_tag_list.append(slice_t[f"{tag_name}"])

            self.tag_dict_main[f"{tag_name}"] = super_tag_list

        
        self.dataframe = pd.DataFrame.from_dict(self.tag_dict_main)

        print(f"Patient chunk length: {len(self.dataframe.index)}")

        self.col_names = list(self.dataframe.columns.values)
        self.patient_ids = self.dataframe['Patient ID'].unique()
        return None
    
    def selector(self, col, func):

        bool_array = self.dataframe[self.col_names[col]].apply(lambda x: func(x))
        self.dataframe = self.dataframe[bool_array]
        self.patient_ids = self.dataframe['Patient ID'].unique()
        return None 
    
    def rescaler_init(self):
        '''
        Load full set of data to find image limits for rescaling

        Uses the following tags:

        (0x0020, 0x0032) Image Position (Patient) (x, y, z coordinates for the left-hand corner)
        (0x0020, 0x0037) Image Orientation (Patient) (Direction of cosines of the first row and the first column)
        (0x0028, 0x0030) Pixel Spacing (x, y) - physical distance between pixel centres, [row spacing, column spacing]
        '''

        scale_file_dirs = glob(f"{self.main_path}/*/*/*/*.dcm")
        junk_x_space = []
        junk_y_space = []
        mean_kvp_arr = []
        scale_array = []
        opp_scale_array = []
        for slice_filepath in tqdm(scale_file_dirs):
                slice_file = pydicom.dcmread(slice_filepath)              
                
                if slice_file[0x0008,0x0060].value == "CT": 
                    orientation = slice_file[0x0020,0x0037].value
                    if slice_file[0x0020, 0x0032].value[2] >= -125 and slice_file[0x0020, 0x0032].value[2] <= -135:
                        continue
                    else:
                        x_space = slice_file[0x0028,0x0030].value[0]
                        x_disp = slice_file[0x0020, 0x0032].value[0] + orientation[0]*x_space*512
                        y_space = slice_file[0x0028,0x0030].value[1]
                        y_disp = slice_file[0x0020, 0x0032].value[1] + orientation[4]*y_space*512

                        opp_scale_array.append([x_disp, y_disp])
                        scale_array.append(slice_file[0x0020, 0x0032].value[:2])
                        mean_kvp_arr.append(slice_file[0x0018,0x0060].value)
                        junk_x_space.append(x_space)
                        junk_y_space.append(y_space)


        scale_array = np.array(scale_array)
        opp_scale_array = np.array(opp_scale_array)
        norm_array = []
        for i in range(len(scale_array)):
            norm_array.append(np.linalg.norm(scale_array[i] - opp_scale_array[i]))
        norm_array = np.array(norm_array)
        
        # Mode l2-nor between corners
        vals_diag, indices_diag = np.unique(norm_array, return_index=True)
        self.mode_norm = vals_diag[np.argsort(indices_diag)][0]

        # Mode pixel centre separation
        vals_x, indices_x = np.unique(junk_x_space, return_index=True)
        self.mode_x = vals_x[np.argsort(indices_x)][0]*512        

        vals_y, indices_y = np.unique(junk_y_space, return_index=True)
        self.mode_y = vals_y[np.argsort(indices_y)][0]*512     

        vals_scale, indices_scale = np.unique(scale_array, return_index=True)
        self.mode_scale = vals_scale[np.argsort(indices_scale)][0]

        # Mean KVP of images
        self.mean_kvp = np.array(mean_kvp_arr).mean()
        print(f"Mean KVP: {self.mean_kvp}")

        return None
    
    def rescaler(self, data_element):
        pixel_array = data_element[self.col_names[-1]]
        position = data_element[self.col_names[1]]
        pixel_spacing = np.array(data_element[self.col_names[5]])
        orientation = np.array(data_element[self.col_names[2]])

        # Flip relevant images
        if orientation[0] == -1 and orientation[4] == -1:
            pixel_array = np.rot90(pixel_array, 2)
            position = -1*np.array(position[:2])

        # Rescale slope and intercept
        intercept = data_element[self.col_names[6]]
        slope = data_element[self.col_names[7]]

        #print(f"Intercept: {intercept}")
        #print(f"Slope: {slope}")

        if intercept <= -1000:
            pixel_array[pixel_array == -2000] = 0

        pixel_array = slope*pixel_array + intercept

        # Resizing the images
        x_disp = position[0] + pixel_spacing[0]*512
        y_disp = position[1] + pixel_spacing[1]*512
        position = np.array(position[:2])
        opp_position = np.array([x_disp, y_disp])

        #print(f"Position: {position}")
        #print(f"Opp. Position: {opp_position}")

        diag_val = np.linalg.norm(position - opp_position)
        xy_displacement = pixel_spacing*512

        x_ratio = self.mode_x/xy_displacement[0]
        y_ratio = self.mode_y/xy_displacement[1]

        #print(f"X Ratio: {x_ratio}")
        #print(f"Y Ratio: {y_ratio}")       

        if diag_val > (self.mode_norm + 5):
            #print("Subsampling:")

            region_selected_X = round(512*(1-x_ratio)/2)
            region_selected_Y = round(512*(1-y_ratio)/2)

            #print(f"Region selected X: {region_selected_X}")
            #print(f"Region selected Y: {region_selected_Y}")

            pixel_array = pixel_array[region_selected_X:512 - region_selected_X, region_selected_Y:512 - region_selected_Y]

        elif diag_val < (self.mode_norm - 5):
            #print("Supersampling:")
            dims_X = round(512*x_ratio)
            dims_Y = round(512*y_ratio)    

            if dims_X % 2 != 0 or dims_Y % 2 != 0:
                dims_X += 1
                dims_Y += 1

            #print(f"New Image Dims X: {dims_X}")
            #print(f"New Image Dims Y: {dims_Y}")

            temp_array = np.zeros((dims_X, dims_Y))    

            region_selected_X = round((dims_X - 512)/2)
            region_selected_Y = round((dims_Y - 512)/2)

            #print(f"Region selected X: {region_selected_X}")
            #print(f"Region selected Y: {region_selected_Y}")

            temp_array[region_selected_X:dims_X - region_selected_X, region_selected_Y:dims_Y - region_selected_Y] = pixel_array[:, :]

            pixel_array = temp_array

        centre = np.floor((512-1)/2)

        # Set all out of range pixels back to -1024 
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                arg_dist = round(np.sqrt((centre - i)**2 + (centre - j)**2))
                if arg_dist > centre:
                    pixel_array[i, j] = -1024

        # Normalize image to 0 -> 1 range
        #print(f"Image Maximum: {pixel_array.max()}")
        #print(f"Image Minimum: {pixel_array.min()}")
        MIN = pixel_array.max()
        MAX = pixel_array.min()
        pixel_array = pixel_array/(MIN - MAX)

        # Rescale image to 256x256 pixels with an anti-aliasing filter
        im_array = Image.fromarray(np.float32(pixel_array))
        im_resized = im_array.resize((256, 256))
        pixel_array = np.array(im_resized)

        pixel_array = pixel_array*(MAX - MIN)
        return pixel_array

    def rescale_prior(self):
        
        holder = pd.DataFrame(columns=["Index", f"{self.col_names[-1]} (Norm)"])

        for patient in self.patient_ids:
            is_patient = (self.dataframe['Patient ID'] == patient)
            patient_array = self.dataframe[is_patient == True]

            temp_array = []
            for i in range(patient_array.shape[0]):
                self.rescaler(data_element = patient_array.iloc[i])

            patient_array = np.stack(temp_array, axis=0)

            is_patient_idx = np.array(np.where(is_patient)).ravel()

            temp = pd.DataFrame(columns=["Index", f"{self.col_names[-1]} (Norm)"]) #, columns=["Index", f"{self.col_names[-1]} (Norm)"]

            temp[f"{self.col_names[-1]} (Norm)"] = patient_array.tolist()
            temp["Index"] = is_patient_idx.tolist()

            holder = pd.concat([temp, holder])
            
        holder.set_index("Index", inplace = True)
        holder.sort_index(axis=0, inplace = True)
        self.dataframe = self.dataframe.join(holder)
        self.col_names = list(self.dataframe.columns.values)
        return None
    
    def prior_loader(self, chunk_size, tag_list=None): #col, func, 
        first_moments_list = []
        second_moments_list = []
        chunk_size_list = []
        curr_total_size = 0
        num_chunks = np.ceil(len(os.listdir(self.main_path)) / chunk_size).astype(int)

        first_moment_path[:] = 0.0
        second_moment_path[:] = 0.0
        for i in range(num_chunks):
            self.chunk_loadin(chunk_size, chunk_ind=i, tag_list=tag_list)

            self.selector(col=1, func=select_slice)
            self.rescale_prior()

            self.actual_chunk_size = len(self.patient_ids)
            chunk_size_list.append(self.actual_chunk_size)
            curr_total_size += self.actual_chunk_size

            patient_intensity_avg = []

            for patient in self.patient_ids:
                is_patient_filtered = (self.dataframe['Patient ID'] == patient)
                patient_slices = self.dataframe.loc[is_patient_filtered, self.col_names[-1]]

                patient_slices = np.stack(patient_slices.to_numpy()).mean(axis=0)

                patient_intensity_avg.append(patient_slices.ravel())

            patient_intensity_avg = np.array(patient_intensity_avg)
            print(f"Chunk Data Shape: {patient_intensity_avg.shape}")

            print(f"Current chunk index: {i}")
            print(f"Total size: {curr_total_size}")
            print(f"Current chunk size: {self.actual_chunk_size}")            

            # First Moment Update
            temp_first_moment = np.mean(patient_intensity_avg, axis=0)

            first_moment_path[:] = ((curr_total_size - self.actual_chunk_size)*first_moment_path[:] + self.actual_chunk_size*temp_first_moment[:])/curr_total_size
            
            print(f"Current first mom. range: {first_moment_path.min()} -> {first_moment_path.max()}") 

            # Second Moment Update
            print("Second Moment Update Progress:")
            for sub_i in tqdm(range(512)):
                sec_st_i = 512*sub_i
                sec_en_i = 512*sub_i + 512

                for sub_j in range(512):
                    sec_st_j = 512*sub_j
                    sec_en_j = 512*sub_j + 512

                    N = patient_intensity_avg.shape[0]
                    temp_second_moment = (1/N)*(patient_intensity_avg.T[sec_st_i : sec_en_i] @ patient_intensity_avg[:, sec_st_j : sec_en_j])
                    temp_second_moment = ((curr_total_size - self.actual_chunk_size)*second_moment_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] + self.actual_chunk_size*temp_second_moment)/curr_total_size
                    second_moment_path[sec_st_i : sec_en_i,  sec_st_j : sec_en_j] = temp_second_moment

            print(f"Current second mom. range: {second_moment_path.min()} -> {second_moment_path.max()}")    
            #break
                
        return None

def select_slice(val):
    bool = (val[2] < -125.0) and (val[2] > -135.0)
    return bool



main_path = "D:/Studies/MEng_Project/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI"

'''
List of tags used:
1. (0x0020, 0x0032) Image Position (Patient) (x, y z coordinates for the upper-left corner)
2. (0x0020, 0x0037) Image Orientation (Patient) (Direction of cosines of the first row and the first column)
3. (0x0028, 0x0010) Number of Rows
4. (0x0028, 0x0011) Number of Columns
5. (0x0028, 0x0030) Pixel Spacing (x, y) - physical distance between pixel centres, [row spacing, column spacing]
6. (0x0028, 0x1052) Rescale Intercept - b in Output units = m*SV+b
7. (0x0028, 0x1053) Rescale Slope - m in Output units = m*SV+b
-1. (0x7FE0, 0x0010) Pixel Data (always loaded in at the end)
'''

tag_list = [[0x0020, 0x0032], 
            [0x0020, 0x0037],  
            [0x0028, 0x0010], 
            [0x0028, 0x0011],
            [0x0028, 0x0030], 
            [0x0028, 0x1052],
            [0x0028, 0x1053],
            [0x7FE0, 0x0010]]


dataset = Dataset(main_path=main_path)
dataset.chunk_loadin(chunk_size=10, tag_list=tag_list)
dataset.selector(col=1, func=select_slice)
empty_array = []
for i in range(dataset.dataframe.shape[0]):
    empty_array.append(dataset.rescaler(data_element=dataset.dataframe.iloc[i]))
empty_array = np.array(empty_array)

imageio.mimsave('D:/Studies/MEng_Project/test_file20.gif', empty_array, duration=1000)

#print(np.unique(x_array))
#print(np.unique(y_array))
#fig, ax = plt.subplots(nrows=1, ncols=2)

#ax[0].hist(x_array, bins=100, color='b')
#ax[1].hist(y_array, bins=100, color='b')

#ax.scatter(array[:, 0], array[:, 1], edgecolors='k', c='b') #, cmap='Greys', aspect='auto'
#ax.scatter(opp_array[:, 0], opp_array[:, 1], edgecolors='k', c='r') 

#plt.colorbar(im, ax=ax)
#plt.show()

#test_load = dataset.prior_loader(chunk_size=75, tag_list=tag_list)
#dataset.chunk_loadin(chunk_size=10, tag_list=tag_list)
#dataset.selector(col=1, func=select_slice)
#print(dataset.dataframe['Image Position (Patient)'].apply(lambda x: select_slice(x)))
'''
print(dataset.col_names)
print(dataset.patient_ids)
print(dataset.df_filtered)
'''
#print(test_load.max())
'''
fig, ax = plt.subplots()
im = ax.imshow(test_load.reshape([512, 512]), cmap='Greys', aspect='auto')
plt.colorbar(im, ax=ax)
plt.show()
'''
#imageio.mimsave('D:/Studies/MEng_Project/test_file20.gif', test_load*1000, duration=1000)

