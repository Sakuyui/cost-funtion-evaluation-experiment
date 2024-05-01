import numpy as np
import os, sys
import pandas as pd
from numpy.random import rand
from scipy.sparse import spdiags,linalg,eye
import multiprocess
from scripts.utils import *
from scripts.evaluation import *

runtime_config = {
    'generate_random_samples':{
        'enabled': False,
        'sizes': [100, 200, 500, 1000, 2000, 5000],
        'repeats': [50, 50, 50, 50, 50, 50],
        'total_neuron_count': 1500,
        'path': './model_samples/'
    },
    'evaluation_config':{
        'report_base_path': './evaluation_reports',
        'evaluation_items': [
            {
                'example-type': 'metropolis',
                'example-size': [100, 200, 500, 1000, 2000], #, 5000],
                'total_neuron_count': 1500,
                'trainning_method': 'observe',
                'desc': 'metropolis-observe'
            },
            {
                'example-type': 'random',
                'example-size': [100, 200, 500, 1000, 2000], #, 5000],
                'total_neuron_count': 1500,   
                'repeat': 10,
                'trainning_method': 'observe',
                'desc': 'random-observe'        
            },
            {
                'example-type': 'metropolis',
                'example-size': [100, 200, 500, 1000, 2000], #, 5000],
                'total_neuron_count': 1500,
                'trainning_method': 'pp',
                'desc': 'metropolis-pp'       
            },
            {
                'example-type': 'random',
                'example-size': [100, 200, 500, 1000, 2000], #, 5000],
                'total_neuron_count': 1500,     
                'repeat': 10,
                'trainning_method': 'pp',
                'desc': 'random-pp'          
            }
        ]
    }
}

def _get_H_J(evaluation_item):
    if evaluation_item['trainning_method'] == 'observe':
        return observed_si, observed_sisj
    
    return None, None
    

def _evaluate(evaluation_item):
    costs = []
    H, J = _get_H_J(evaluation_item)
    if H == None or J == None:
        raise ValueError
    
    
    
    with multiprocess.Pool(MAX_PROCESSORS) as pool:
        results = [pool.apply_async(calculate_configuration_cost_with_fixed_size, fixed_size) for fixed_size in map(lambda x: [x, H, J, samples], range(0, 1200))]
        costs = [result.get() for result in results]

    return (pd.DataFrame({'d1': d1, 'd2':d2, 'd3':d3, 'd4':d4, 'costs': costs}).corr())

    pass

# Evaluation

def evaluate_main(evaluation_config):
    report_save_base_path = evaluation_config['report_base_path']
    for index, evaluation_item in enumerate(evaluation_config['evaluation_items']):
        print("---------- Begin Evaluate %d Evaluation Item. ------------" % (index + 1))
        report : pd.DataFrame = _evaluate(evaluation_item)
        report.to_csv(os.path.join(report_save_base_path, evaluation_item['desc'] + ".csv"))
    

if 'generate_random_samples' in runtime_config and 'enabled' in runtime_config['generate_random_samples'] and runtime_config['generate_random_samples']['enabled']:
    print(" ------- Generate Random Examples -------")
    generate_random_samples(configuration=runtime_config['generate_random_samples'])

print(" ------- Load records -------")
records = load_all_record()
d1 = calc_observed_item_average('energy-used-by-chips', 'joules', records)
d2 = calc_observed_item_average('energy-used-by-chips', 'secs', records)
d3 = calc_observed_item_average('energy-used-by-packet-transmissions', 'joules', records)
d4 = calc_observed_item_average('energy-used-by-packet-transmissions', 'secs', records)


# Definition of data source configuration
base_spike_record_file_path = os.path.join("./spike_data/")
population_names = ["neo_exe_cells", "neo_inh_cells"]
observed_si, observed_sisj = get_observed_si_sisj(population_names, base_spike_record_file_path)

# load sample set
metropolis_examples = np.load(os.path.join(model_samples_base_path, metropolis_examples_file_path))
# random_examples = np.load(os.path.join(model_samples_base_path, random_generate_examples_file_path))


evaluate_main(runtime_config['evaluation_config'])