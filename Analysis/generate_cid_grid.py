# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

import os
import sys
import argparse

import numpy as np
import pandas as pd

from cellsegmentationtracker import CellSegmentationTracker

user = os.environ.get("USERNAME")
projects_dir = os.environ['Projects']
project_dir = os.path.join(projects_dir, 'MonolayerTransition',)

working_dir = os.path.join(project_dir, 'Analysis',)
data_dir = os.path.join(project_dir, 'data',)
data_folders = os.listdir(data_dir)


### FUNCTIONS ----------------------------------------------------------------------------------

def construct_arr_from_dataframe(grid_df, target_feature='cell_number', verbose=False):
    """
    Construct a 3D numpy array from a grid statistics dataframe.

    :param grid_df: pandas DataFrame with grid statistics, must contain columns:
                    'x_center', 'y_center', 'Frame', and target_feature
    :param target_feature: str, name of the feature column to populate the array
    :param verbose: bool, if True, print debug information

    :return: lattice_array: 3D numpy array of shape (Mx, My, N) where Mx, My are spatial grid dimensions
             and N is the number of time frames.
    """

    # Get dimensions
    Mx = len(grid_df['x_center'].unique())
    My = len(grid_df['y_center'].unique())
    N = int(grid_df['Frame'].max()) + 1  # number of time points

    # Get spatial bounds from the grid
    x_min, x_max = grid_df['x_center'].min(), grid_df['x_center'].max()
    y_min, y_max = grid_df['y_center'].min(), grid_df['y_center'].max()

    if verbose:
        print(f"Grid dimensions: {Mx} x {My} x {N}")
        print(f"Spatial bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    # Initialize the array
    xvals = grid_df['x_center'].unique()
    yvals = grid_df['y_center'].unique()
    lattice_array = np.zeros((Mx, My, N), dtype=np.float32)

    # Map dataframe values to array indices
    for idx, row in grid_df.iterrows():
        # Convert spatial coordinates to grid indices (0 to M-1)
        x_idx = int(np.round((row['x_center'] - x_min) / (x_max - x_min) * (Mx - 1)))
        y_idx = int(np.round((row['y_center'] - y_min) / (y_max - y_min) * (My - 1)))
        t_idx = int(row['Frame'])
        
        # Ensure indices are within bounds
        x_idx = np.clip(x_idx, 0, Mx - 1)
        y_idx = np.clip(y_idx, 0, My - 1)
        t_idx = np.clip(t_idx, 0, N - 1)

        lattice_array[x_idx, y_idx, t_idx] = row[target_feature]
    return lattice_array, xvals, yvals


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--data_set', type=str, default=None, 
                        help='Name of data set folder inside data_dir. If None, use all folders in data_dir. \
                            allowed values: cn03, cytoD, low_density, high_density')
    parser.add_argument('-tf', '--target_feature', type=str, default='cell_number',  
                        help='Feature to extract from grid statistics')
    parser.add_argument('-N', '--Ngrid', type=int, default=50)
    parser.add_argument('-v', '--verbose', type=str2bool, default=False)
    args = parser.parse_args()

    target_feature=args.target_feature
    data_set = args.data_set
    verbose = args.verbose

    xbounds = (200, 1200) # in physical units
    ybounds = (200, 1200) # in physical units

    unit_conversion_dict= {
                    'frame_interval_in_physical_units': 600,
                    'physical_length_unit_name': r'pixels',
                    'physical_time_unit_name': 's',}
    grid_params = {
            'grid_boundaries': [xbounds, ybounds],
            'Ngrid': args.Ngrid,
            'include_features': ['Area', 'Perimeter',
                                'Circularity','Solidity',
                                'Shape index', 'Velocity_X',
                                'Velocity_Y'],
        }

    if data_set is None:
        folder_names = data_folders
    else:
        folder_names = [data_set + '_islands']

    print("Processing data folders:", folder_names)
        
    for folder_name in folder_names:
        # for cn03/cytoD, first 6 frames are prior treatment, the rest post
        if folder_name.startswith('cn') or folder_name.startswith('cytoD'):
            island_num = 5
            img_suffix = '_prior' # '_prior' or '_post'
        else:
            island_num = 1
            img_suffix = ''

        data_folder = os.path.join(data_dir, folder_name, f'island{island_num:02d}')
        img_name = f'c2_crop{img_suffix}'
 
        cst_params = {'output_folder_path': data_folder,}
        
        # Initialize CellSegmentationTracker object
        cst = CellSegmentationTracker(
            **cst_params,
            unit_conversion_dict = unit_conversion_dict)
        
        # Load spots, tracks, and edges dataframes
        cst.spots_df = pd.read_csv(os.path.join(data_folder, f'{img_name}_spots.csv'))

        # Calculate grid statistics dataframe if not already done
        name = f'grid_df_N{args.Ngrid}'
        csv_path = os.path.join(data_folder, f'{name}.csv')
        if os.path.exists(csv_path):
            if verbose:
                print(f"Loading existing grid statistics from {csv_path}")
            grid_df = pd.read_csv(csv_path)
        else:
            grid_df = cst.calculate_grid_statistics(**grid_params, name=name)
            
        # construct lattice array from dataframe
        lattice_array, xvals, yvals = construct_arr_from_dataframe(
                                    grid_df, 
                                    target_feature=target_feature,
                                    verbose=verbose)
        if verbose:
            print(f"Number of non-zero elements: {np.count_nonzero(lattice_array)}")
            print(f'fraction of non-zero elements: {np.count_nonzero(lattice_array) / lattice_array.size:.4f}')

        # save the array, along with xvals and yvals as npz
        np.savez_compressed(os.path.join(data_folder, f'{target_feature}_N{args.Ngrid}arr.npz'),
                             lattice_array=lattice_array, xvals=xvals, yvals=yvals)
      
        if verbose:
            print(f"Saved lattice_array to {os.path.join(data_folder, f'{target_feature}_N{args.Ngrid}arr.npz')}")


if __name__ == '__main__':
    main()

   