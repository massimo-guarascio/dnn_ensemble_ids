#import libraries
import os
import pandas as pd
import numpy as np
import random as rn
from scipy.io import arff
#SEED
seed = 23071982
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(seed)

import sys

def readData(path, delimiter, decimal):
    "READ SINGLE FILE"
    df=pd.read_csv(path, delimiter=delimiter, decimal=decimal, engine="c", header = 0)
    return df

def check_good(value):
    return 0.0 if value == "BENIGN" else 1.0

def port_category(value):
    if value < 1024 :
        return "well_known"
    if value < 49152:
        return "registered"
    return "dynamic"


#MAIN
if __name__ == "__main__":

    # --- PARAMETERS ---
    # Tuesday , Wednesday, Thursday , Friday
    dataset_path = "..."
    output_path = "..."

    # other parameters
    delim = ","
    decimal = "."

    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)

    print("--- LOADED DATASET ---")

    # PRINT COLUMN NAMES
    print(data.columns)

    # PRINT HEAD OF THE DATASET
    print(data.head(3))

    #PRINT CLASS DISTRIBUTION
    print(data["Label"].value_counts())

    data["port_type"] = data["Destination_Port"].apply(port_category)

    data["class"] = data["Label"].apply(check_good)

    data = data.drop("Label", 1)

    #PRINT CLASS DISTRIBUTION
    print("AFTER")
    print(data["class"].value_counts())
    print(data["port_type"].value_counts())

    # Write
    data.to_csv(output_path, sep=delim, decimal=decimal, index=False)

