# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

from doctest import debug
import io
import sys
import os
import pickle
import warnings
import time
import argparse
from multiprocessing.pool import Pool as Pool
from pathlib import Path

import numpy as np

notebook_path = Path().resolve()
parent_dir = notebook_path.parent
sys.path.append(str(parent_dir))
sys.path.append('..\\ComputableInformationDensity')

from ComputableInformationDensity.cid import CID, CID_old
from ComputableInformationDensity.computable_information_density import cid

user = os.environ.get("USERNAME")
projects_dir = os.environ['Projects']
project_dir = os.path.join(projects_dir, 'MonolayerTransition',)

working_dir = os.path.join(project_dir, 'Analysis',)
data_dir = os.path.join(project_dir, 'data',)
data_folders = os.listdir(data_dir)


### FUNCTIONS ----------------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

### MAIN ---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--data_set', type=str, default=None, 
                        help='Name of data set folder inside data_dir. If None, use all folders in data_dir. \
                            allowed values: cn03, cytoD, low_density, high_density')
    parser.add_argument('-tf', '--target_feature', type=str, default='cell_number',  
                        help='Feature to extract from grid statistics'),
    parser.add_argument('-cg', '--coarse_graining_box_length', type=int, default=1),
    parser.add_argument('-nt', '--Nframes_per_cube', type=int, default=1)
    parser.add_argument('-N', '--Ngrid', type=int, default=50)
    parser.add_argument('-v', '--verbose', type=str2bool, default=False)
    parser.add_argument('-d', '--debug', type=str2bool, default=False)
    args = parser.parse_args()

    Ngrid = args.Ngrid
    target_feature=args.target_feature
    data_set = args.data_set
    verbose = args.verbose

    # CID parameters
    nframes_per_cube = args.Nframes_per_cube
    coarse_graining_box_length = args.coarse_graining_box_length
    overlap = 0 # overlap between cubes in time direction
    njumps_between_frames = 1 # use every nth frame to reduce temporal correlation
    defect_dtype = int
    dtype = np.uint8
    cid_mode = 'lz77'

    
    if data_set is None:
        folder_names = data_folders
    else:
        folder_names = [data_set + '_islands']

    print("Processing data folders:", folder_names)
        
    for folder_name in folder_names:
        # for cn03/cytoD, first 6 frames are prior treatment, the rest post
        if folder_name.startswith('cn') or folder_name.startswith('cytoD'):
            island_num = 5
            img_suffix = '_post' # '_prior' or '_post'
        else:
            island_num = 1
            img_suffix = ''

        data_folder = os.path.join(data_dir, folder_name, f'island{island_num:02d}')
        npz_path = os.path.join(data_folder, f'{target_feature}_N{Ngrid}arr.npz')

        if verbose:
            print("extracting data from folder:\n", data_folder)

        # load npz file
        with np.load(npz_path) as data:
            lattice_array = data['lattice_array'].astype(np.int8)
            xvals = data['xvals']
            yvals = data['yvals']

        nshuffle = os.cpu_count() 
        sequential = True if nframes_per_cube == 1 else False

        # Set params
        num_frames_max = lattice_array.shape[-1]
        window_size = lattice_array.shape[0]  # assuming square lattice
        LX = lattice_array.shape[0]

        Nframes = lattice_array.shape[-1]
        nframes_to_analyze = min(num_frames_max, Nframes) 

        LX_cg = LX // coarse_graining_box_length
        compression_factor = LX // window_size # number of partitions along one dimension
        npartitions = compression_factor**2
        lx_window_cg = window_size // coarse_graining_box_length

        hyperwindow_shape = (nframes_per_cube, window_size, window_size) if not sequential else (window_size, window_size)
        hyperwindow_shape_cg = (nframes_per_cube, lx_window_cg, lx_window_cg) if not sequential else (lx_window_cg, lx_window_cg)
        hyperwindow_dim = len(hyperwindow_shape_cg)
        nzorder_permutations = int(np.math.factorial(hyperwindow_dim))
        nshuffle = max(nshuffle, nzorder_permutations)

        save_suffix = f'{target_feature}_ng{Ngrid}_nt{nframes_per_cube}cg{coarse_graining_box_length}'

        time_subinterval = nframes_per_cube - overlap
        ncubes = 1 + int(((nframes_to_analyze - nframes_per_cube) / time_subinterval))
        first_frame_idx = (Nframes - nframes_to_analyze) + (nframes_to_analyze - nframes_per_cube - ((ncubes - 1) * time_subinterval))

        ## print summary of parameters
        if verbose:
            print("\n=== CID Calculation Parameters Summary ===")
            print(f"Data folder: {data_folder}")
            print(f"Number of frames to analyze: {nframes_to_analyze} out of {Nframes}")
            print(f"Window size: {window_size}x{window_size}")
            print(f"Frames per cube: {nframes_per_cube}")
            print(f"npartitions: {npartitions}")
            print(f"Coarse graining box length: {coarse_graining_box_length}")
            print(f"Number of cubes to analyze: {ncubes}")
            print(f"Hyperwindow shape (coarse-grained): {hyperwindow_shape_cg}")

        # Initialize CID 
        defect_grid = np.swapaxes(lattice_array, 0, 2)
        CID_obj = CID(dim=hyperwindow_dim, 
                        data_shape=hyperwindow_shape_cg, 
                        nshuff=nshuffle, 
                        mode=cid_mode, 
                        ordering='zcurve', 
                        verbose=False)

        cid_arr_full = np.nan * np.ones((ncubes, npartitions, nzorder_permutations))
        cid_arr = np.nan * np.ones((ncubes, npartitions, 2))
        cid_shuffle_arr = np.nan * np.ones_like(cid_arr)
        cid_frac_arr = np.nan * np.ones_like(cid_arr)

        # Compute CID for null cube
        null_cube = np.zeros(hyperwindow_shape_cg, dtype=dtype)
        cid_min = cid(null_cube.flatten())

        for j in range(ncubes):
            for i in range(npartitions):
                if npartitions > 1:
                    x_start = (i % compression_factor) * lx_window_cg
                    y_start = (i // compression_factor) * lx_window_cg
                    data = defect_grid[j * time_subinterval:(j+1)*time_subinterval+overlap, x_start:x_start+lx_window_cg, y_start:y_start+lx_window_cg]
                else:
                    data = defect_grid[j * time_subinterval:(j+1)*time_subinterval+overlap, :, :]
                    print(data.shape)
                if data.sum() == 0:
                    cid_av, cid_std = cid_min, 0
                    cid_shuff = cid_min, 0
                    cid_vals = cid_min * np.ones(cid_arr_full.shape[-1])
                else:
                    cid_av, cid_std, cid_shuff, cid_vals = CID_obj(data)          
                    
                cid_arr[j, i, 0] = cid_av
                cid_arr[j, i, 1] = cid_std
                cid_shuffle_arr[j, i, :] = cid_shuff
                cid_arr_full[j, i, :] = cid_vals
            if verbose: 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    print(f"For {folder_name}: av CID, CID_shuff, frac: {cid_av:.5f}, {cid_shuff[0]:.5f}, {cid_av/cid_shuff[0]:.5f} +/- {cid_std/cid_shuff[0]:.5f}")

        cid_frac_arr[..., 0] = cid_arr[..., 0] / cid_shuffle_arr[..., 0]
        cid_frac_arr[..., 1] = cid_frac_arr[..., 0] * np.sqrt( (cid_arr[..., 1]/cid_arr[..., 0])**2 + (cid_shuffle_arr[..., 1]/cid_shuffle_arr[..., 0])**2 )

        # squeeze if only one partition
        if npartitions == 1:
            cid_arr = np.squeeze(cid_arr, axis=1)
            cid_shuffle_arr = np.squeeze(cid_shuffle_arr, axis=1)
            cid_frac_arr = np.squeeze(cid_frac_arr, axis=1)        

        if not debug:
            output_path = os.path.join(data_folder, f'cid_{target_feature}')
            np.savez_compressed(os.path.join(output_path, f'cid{save_suffix}.npz'), 
                                cid=cid_arr, 
                                cid_shuffle=cid_shuffle_arr, 
                                cid_frac=cid_frac_arr,
                                cid_full=cid_arr_full,)
            if verbose:
                print(f"CID results saved to {output_path}/cid{save_suffix}.npz")
    
if __name__ == '__main__':
    main()

   