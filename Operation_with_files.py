import numpy as np
import matplotlib.pyplot as plt
import h5py
import os


# This class was implemented to efficiently manage and visualize simulation results.
# The h5py package is utilized to persist data across different stages of the iterative process.

class Operation_with_files:

    # Class constructor initializing the target directory and the full filepath configuration.
    def __init__(self, filename):
        self.Files_Path = 'Files'
        self.filename = os.path.join(self.Files_Path, filename)

    # Retrieves data from a specific dataset within the HDF5 container by its name.
    def Read_file(self, name):
        with h5py.File(self.filename, 'r') as f:
            data = f[name][:]
        return data

    # Writes data into the specified dataset. If a dataset with the same name
    # already exists, it is overwritten with the new array values.
    def Write_to_file(self, name, Data):
        with h5py.File(self.filename, 'a') as f:
            if name in f:
                del f[name]
            f.create_dataset(name, data=Data)