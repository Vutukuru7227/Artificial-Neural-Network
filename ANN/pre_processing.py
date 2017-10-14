import csv
import sys
import numpy
import pandas as pd

def get_input_data(input_path, output_path):

    #with open(input_path, 'rt') as csv_file:
    #    next(csv_file)
    #    reader = csv.reader(csv_file)
    #    raw_data = list(reader)
    raw_data = pd.read_csv(input_path, header=0)
    raw_data.head(3)

    return raw_data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: pre_processing.py <input path> <output path>")
    else:
        # TODO: Function call to fetch the data and feature labels
        raw_data = get_input_data(sys.argv[1], sys.argv[2])
        print("Started")
        #with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(raw_data)
        #Get rid of null values
        raw_data.dropna()
        print(raw_data)
        #Convert values to numeric
        raw_data.apply(pd.to_numeric)
            #pd.to_numeric(raw_data, "raise", None)
        print(raw_data)
        #Normalize
            #print(raw_data.mean())
            #print(raw_data.max())
            #print(raw_data.min())
        raw_data = (raw_data - raw_data.mean()) / (raw_data.max() - raw_data.min())
        print(raw_data)
        print("Finished")