from sys import argv
from os import makedirs
import json
import numpy as np
from numpy.lib.polynomial import polyval
import pandas as pd
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
PRINT_START_TIME = 10

def read_results_file(results_fp):
    with open(results_fp, 'r') as f:
        results = json.load(f)
    return(results)

class FrameExpansionCalibrator(object):
    def __init__(self, inputfile):
        with open(inputfile, 'r') as f:
            results_all = json.load(f)

        self.metadata = results_all['metadata']
        self.temp_data = pd.read_json(json.dumps(results_all['temp_data']),
            orient='index')
        self.cold_mesh = results_all['cold_mesh']
        self.hot_mesh = results_all['hot_mesh']

if __name__ == "__main__":
    args = argv[1:]
    # print(args)
    for arg in args:
        dataset_name = arg.strip('.\\').replace('.json', '')
        print("Analyzing file: %s" % dataset_name)
        results = read_results_file(arg)
        step_dist = results['metadata']['z_axis']['step_dist']
        username = results['metadata']['user']['id']
        dataset_timestamp = results['metadata']['user']['timestamp']
        temp_data = pd.read_json(json.dumps(results['temp_data']), orient='index')
        temp_data.reset_index(inplace=True)
        temp_data.rename(columns= {'index':'timestamp'}, inplace=True)
        temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'], 
                                                format="%Y/%m/%d %H:%M:%S")
        output_path = "%s/%s/" % (username, dataset_timestamp)
        makedirs(output_path, exist_ok=True)
        temp_data['elapsed_min'] = temp_data['timestamp'].diff().dt.total_seconds()\
            .cumsum().fillna(0).div(60)

        temp_data_mean = temp_data.groupby('sample_index').mean()
        temp_data_mean['delta_z'] = temp_data_mean['mcu_z']\
            .transform(lambda x: (x-x.iloc[0])*step_dist)

        temp_data_sd = temp_data.groupby('sample_index').std()
        temp_data_sd['delta_z'] = temp_data_sd['mcu_z']\
            .transform(lambda x: np.sqrt(x**2 + x.iloc[0]**2)*step_dist)

        filtered_mean = temp_data_mean[(temp_data_sd['mcu_z'] < 2)
            & (temp_data_mean['bed_target'] > 0)
            & (temp_data_mean['frame_temp'] < temp_data_mean['frame_temp'].max()-0.3)]


        filter_mean_print = filtered_mean.loc[filtered_mean['elapsed_min'] > PRINT_START_TIME]
        m, c = np.polyfit(filter_mean_print['frame_temp'],
                          filter_mean_print['delta_z'], 
                          1)

        filter_mean_print['fit_z'] = np.polyval([m, c], filter_mean_print['frame_temp'])

        plt.figure(1, (6,6))
        plt.scatter('delta_z', 'frame_temp', c='frame_temp',  cmap='inferno', data=filter_mean_print)
        plt.axline((filter_mean_print['fit_z'].min(),
                    filter_mean_print['frame_temp'].max()
                    ),
                    slope=1/m,
                    linestyle="--",
                    c='black')
        plt.title('%s\nFrame Expansion\nTemperature Coefficient Fitting' %
            dataset_name)
        plt.xlabel('Delta Z [mm]')
        plt.ylabel('Frame Temperature [degC]')
        plt.annotate(text="temp_coeff:\n%.4f mm/K" % (-1*m), xy=(0.6,0.8), xycoords='figure fraction')
        plt.savefig('%stemp_coeff_fitting.png' % output_path)
        plt.close()
