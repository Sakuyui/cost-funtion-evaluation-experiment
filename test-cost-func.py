import numpy as np
import os, sys
import datetime
import pandas as pd
from numpy.random import rand
from scipy.sparse import spdiags,linalg,eye
import random
import re

record_base_path = "./integrated_reports/"
summary_file_format = "s-max-atoms-per-core-%d-trail-%d.rpt"

def read_file(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        data = (f.read())
        f.close()
    return data

def analyze_summary_file(max_atoms_per_core, trails=5):
    joules_regex = r'.* ([0-9]+\.[0-9]+) Joules.*'
    seconds_regex = r'.* ([0-9]+\.[0-9]+) seconds.*'
    milliseconds_regex = r'.* ([0-9]+\.[0-9]+) milliseconds.*'

   
    analyzed_data = dict({})

    def analyze_joules(description):
        joules_match = re.match(joules_regex, description)
        return {
            'joules': (float)(joules_match.group(1))
        }
    def analyze_time(description):
        time_match = re.match(seconds_regex, description)
        if time_match != None:
            return {
                'secs': (float)(time_match.group(1)),
            }
        else:
            time_match = re.match(milliseconds_regex, description)
            return {
                'm_secs': (float)(time_match.group(1)),
            }
    def analyze_general_number(description):
        joules_match = re.match( r'.* ([0-9]+\.[0-9]+) .*', description)
        return (float)(joules_match.group(1))
        
    def analyze_joules_and_seconds(description):
        joules_item = analyze_joules(description)
        time_item = analyze_time(description)
        
        return joules_item | time_item

    analyzed_data['atoms_per_core'] = max_atoms_per_core
    analyzed_data['trail_data'] = []

    for trail_id in range(trails):
        file_path = os.path.join(record_base_path, summary_file_format % (max_atoms_per_core ,trail_id))
        data = read_file(file_path)
        if data == None:
            print("warning %s not exist." % file_path)
            continue
        data = data.split("\n")[3:]

        current_trail_record = dict({})
        
        current_trail_record['trail_id'] = trail_id
    
        current_trail_record['energy-used-by-chips'] = analyze_joules_and_seconds(data[0])
        current_trail_record['energy-used-by-fpgas-entire'] = analyze_joules_and_seconds(data[1])
        current_trail_record['energy-used-by-fpgas-runtime-period'] = analyze_joules_and_seconds(data[2])
        current_trail_record['energy-used-by-outside-route'] = analyze_joules(data[3])
        current_trail_record['energy-used-by-packet-transmissions'] = analyze_joules_and_seconds(data[4])
        current_trail_record['energy-used-by-mapping-process'] = analyze_joules_and_seconds(data[5])
        current_trail_record['energy-used-by-data-generation'] = analyze_joules_and_seconds(data[6])
        current_trail_record['energy-used-by-loading-process'] = analyze_joules_and_seconds(data[7])
        current_trail_record['energy-used-by-data-extraction'] = analyze_joules_and_seconds(data[8])
        current_trail_record['total-simulation-time'] = analyze_time(data[9])
        current_trail_record['total-energy-joules'] = analyze_joules(data[10])
        current_trail_record['avg-kwh'] = analyze_general_number(data[12])

        analyzed_data['trail_data'].append(current_trail_record)

    return analyzed_data

def load_all_record(record_count = 1200):
    all_record = []
    for i in range(record_count):

        all_record.append(analyze_summary_file(i + 1))
    return all_record


print(" ------- Load records -------")
records = load_all_record()

# calculate max fpga differs (factor)
# energy-used-by-chips

def calc_observed_item_average(item_name, observation_type, records, observation_times = 5):
    d1 = list(map(lambda record: record['trail_data'][-1][item_name][observation_type] / observation_times, records))
    print(max(d1), min(d1), max(d1)/ min(d1))
    return d1

d1 = calc_observed_item_average('energy-used-by-chips', 'joules', records)
d2 = calc_observed_item_average('energy-used-by-chips', 'secs', records)
d3 = calc_observed_item_average('energy-used-by-packet-transmissions', 'joules', records)
d4 = calc_observed_item_average('energy-used-by-packet-transmissions', 'secs', records)

def create_configuration(fixed_size, neuron_count):
    config = []
    current_sum = 0
    while current_sum < neuron_count:
        if current_sum < 1200 and current_sum + fixed_size  > 1200:
            config.append(1200 - current_sum)
            current_sum += 1200 - current_sum
        elif current_sum >= 1200 and current_sum < 1500 and current_sum + fixed_size > 1500:
            config.append(1500 - current_sum)
            current_sum += 1500 - current_sum
        else:
            config.append(fixed_size)
            current_sum += fixed_size
    if sum(config) != neuron_count:
        raise ValueError
    return config

def cost(config, H, J, samples, p=0.2):
    if samples.shape[1] != len(H) or len(J.shape) < 2 or samples.shape[1] != J.shape[0] or samples.shape[1] != J.shape[1]:
        raise ValueError
    sample_count = samples.shape[0]
    neuron_count = samples.shape[1]
    cost_sum = 0

    loc_map = [0] * neuron_count

    pos = 0
    core_atoms_count_record = dict({})

    for index, slice_length in enumerate(config):
        core_atoms_count_record[index] = slice_length
        for inner_slice_index in range(slice_length):
            loc_map[pos] = index
            pos += 1

    def no_normalize_p(sample):
        # sample_nor = (sample - (-1)) / 2
        extra_field_factor = -np.sum(H * sample)
        interaction_factor = 0
        neuro_count = sample.shape[0]
        tmp_2d_mat_for_calculate_interaction_factor = sample.T.dot(sample)

        interaction_factor = -np.sum(J * tmp_2d_mat_for_calculate_interaction_factor)
        p =  np.exp(extra_field_factor + interaction_factor)

        return p


    def loc(neuro_index):
        return loc_map[neuro_index]


    for i in range(sample_count):
        p = no_normalize_p(samples[i])
        for j in range(0, neuron_count):
            if samples[i][j] == -1:
                core_id_global = loc(j)
                core_id_inner_chip = core_id_global % 18
                chip_id = core_id_global // 18
                atoms_inner_chip =  core_atoms_count_record[core_id_global]
                c = 10
                cost_sum += p * (atoms_inner_chip * p + c * (len(config) - 1))

    return cost_sum / sample_count

class IsingModel2D(object):
    def __init__(self, neuron_count, J = None, H = None):
        self._J = J if J != None else np.zeros((neuron_count, neuron_count))
        self._H = H if H != None else np.zeros((neuron_count))
        self._neuron_count = neuron_count
        self.configuration = np.zeros((neuron_count))

    def set_J(self, J):
        self._J = J
    def set_H(self, H):
        self._H = H
    def get_J(self):
        return self._J
    def get_H(self):
        return self._H
    def set_J_element(self, i, j, value):
        self._J[i, j] = value
    def set_H_element(self, i, value):
        self._H[i] = value
    def get_neuron_count(self):
        return self._neuron_count

class IsingModelBasedCostFunctionConfigurationExporter(object):
    def __init__(self, base_path, configuration_name) -> None:
        self.base_path = base_path
        self.configuration_name = configuration_name

    def save_ising_model_parameters(self, model: IsingModel2D):
        param_H_file_path = os.path.join(self.base_path, "%s_H.npy" % self.configuration_name)
        param_J_file_path = os.path.join(self.base_path, "%s_J.npy" % self.configuration_name)
        np.save(param_H_file_path, model.get_H())
        np.save(param_J_file_path, model.get_J())

    def save_ising_model_configuration_samples(self, samples):
        samples_file_path = os.path.join(self.base_path, "%s_samples.npy" % self.configuration_name)
        np.save(samples_file_path, samples)

    def save_all_preprocessing_data(self, model, samples):
        self.save_ising_model_parameters(model)
        self.save_ising_model_configuration_samples(samples)


def mcmove(model: IsingModel2D,  beta):
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    N =  model.get_neuron_count()
    for _ in range(1):
        random_selected_index = np.random.randint(0, N)
        s =  model.configuration[random_selected_index]

        # Calculate the energy change if choose to flip the spin.
        ### Let us reflect the formula for energy of a configuration:
        #### E = -\sum_{i = 1, j = 1}^{N - 1,N - 1} .5 * J_{ij} * si * sj -\sum_{i=0}^{N-1} h_i * s_i
        dE = 2 * s * (model.get_H()[random_selected_index] +
                      0.5 * np.sum(model.get_J()[random_selected_index] * model.configuration))

        # In the condition that the energy change less than 0, flip the spin. (A system has a tendency to
        # change to a less energy state, which is more stable.)
        if dE < 0:
            s = - s
            model.configuration[random_selected_index] = s

        elif True if dE == np.inf else rand() < dE == np.exp(-dE * beta):
            s = - s
            model.configuration[random_selected_index] = s

def initilize_ising_model_configuration(model: IsingModel2D):
    model.configuration = np.random.choice([-1, 1], (model.get_neuron_count()))

def initilize_ising_model_by_profiled_si_sisj_ensemble_averages(model: IsingModel2D, si, sisj):
    model.set_H(si)
    model.set_J(sisj)

def calculate_si_m_sisj_m_of_ising_model(model: IsingModel2D, \
                                         equilibration_iteration_count = 2**9,\
                                         sampling_count = 2**13):
    nt = 1
    T = np.linspace(1.53, 3.28, nt);
    accumulated_si_observation = np.zeros((neuron_count))
    accumulated_sisj_observation = np.zeros((neuron_count, neuron_count))

    # for each temperature points
    iT = 1.0/T[0]
    log_interval = equilibration_iteration_count // 100

    for i in range(equilibration_iteration_count):         # equilibrate
        print("equilibration: %d/%d" % (i, equilibration_iteration_count))
        if i % log_interval == 0:
            log(log_file, "equilibration: %d/%d" % (i, equilibration_iteration_count))

        mcmove(model, iT)           # Monte Carlo moves

    print("Finish Equilibration")
    log(log_file, "Finish Equilibration")
    log_interval = sampling_count // 100
    for i in range(sampling_count):
        print("sampling: %d/%d" % (i, sampling_count))
        if i % log_interval == 0:
            log(log_file, "sampling: %d/%d" % (i, sampling_count))

        mcmove(model, iT)
        accumulated_si_observation += model.configuration
        tiled_matrix = np.tile(model.configuration, neuron_count).reshape((neuron_count, neuron_count))
        accumulated_sisj_observation = ((tiled_matrix) + tiled_matrix.T) // 2
        del tiled_matrix


    si = accumulated_si_observation / (sampling_count)
    sisj = accumulated_sisj_observation / (sampling_count)
    return si, sisj

def generate_configuration_samples(model, sample_count = 1e4, equilibration_iteration_count = 1e8):
    # for each temperature points
    configuration_samples = np.zeros((sample_count, neuron_count))
    iT=1.0;
    for i in range(equilibration_iteration_count):         # equilibrate
        mcmove(model, iT)           # Monte Carlo moves

    for i in range(sample_count):
        print("generate %d-th configuration sample" % (i + 1))
        sampled_configuration = mcmove(model, iT)
        configuration_samples[i] = sampled_configuration
    return configuration_samples

def log(logfile, append_content):
	logfile.write("%s\n" % append_content)
	logfile.flush()

config = create_configuration(100, 1500)
H = np.random.randn((1500))
J = np.random.randn((1500 * 1500)).reshape(1500, 1500)
samples = np.random.randint(0, 2, size = (100, 1500))
samples[samples == 0] = -1



class IsingModel2D(object):
    def __init__(self, neuron_count, J = None, H = None):
        self._J = J if J != None else np.zeros((neuron_count, neuron_count))
        self._H = H if H != None else np.zeros((neuron_count))
        self._neuron_count = neuron_count
        self.configuration = np.zeros((neuron_count))

    def set_J(self, J):
        self._J = J
    def set_H(self, H):
        self._H = H
    def get_J(self):
        return self._J
    def get_H(self):
        return self._H
    def set_J_element(self, i, j, value):
        self._J[i, j] = value
    def set_H_element(self, i, value):
        self._H[i] = value
    def get_neuron_count(self):
        return self._neuron_count

class IsingModelBasedCostFunctionConfigurationExporter(object):
    def __init__(self, base_path, configuration_name) -> None:
        self.base_path = base_path
        self.configuration_name = configuration_name

    def save_ising_model_parameters(self, model: IsingModel2D):
        param_H_file_path = os.path.join(self.base_path, "%s_H.npy" % self.configuration_name)
        param_J_file_path = os.path.join(self.base_path, "%s_J.npy" % self.configuration_name)
        np.save(param_H_file_path, model.get_H())
        np.save(param_J_file_path, model.get_J())

    def save_ising_model_configuration_samples(self, samples):
        samples_file_path = os.path.join(self.base_path, "%s_samples.npy" % self.configuration_name)
        np.save(samples_file_path, samples)

    def save_all_preprocessing_data(self, model, samples):
        self.save_ising_model_parameters(model)
        self.save_ising_model_configuration_samples(samples)


def mcmove(model: IsingModel2D,  beta):
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    N =  model.get_neuron_count()
    for _ in range(1):
        random_selected_index = np.random.randint(0, N)
        s =  model.configuration[random_selected_index]

        # Calculate the energy change if choose to flip the spin.
        ### Let us reflect the formula for energy of a configuration:
        #### E = -\sum_{i = 1, j = 1}^{N - 1,N - 1} .5 * J_{ij} * si * sj -\sum_{i=0}^{N-1} h_i * s_i
        dE = 2 * s * (model.get_H()[random_selected_index] +
                      0.5 * np.sum(model.get_J()[random_selected_index] * model.configuration))

        # In the condition that the energy change less than 0, flip the spin. (A system has a tendency to
        # change to a less energy state, which is more stable.)
        if dE < 0:
            s = - s
            model.configuration[random_selected_index] = s

        elif True if dE == np.inf else rand() < dE == np.exp(-dE * beta):
            s = - s
            model.configuration[random_selected_index] = s

def initilize_ising_model_configuration(model: IsingModel2D):
    model.configuration = np.random.choice([-1, 1], (model.get_neuron_count()))

def initilize_ising_model_by_profiled_si_sisj_ensemble_averages(model: IsingModel2D, si, sisj):
    model.set_H(si)
    model.set_J(sisj)

def calculate_si_m_sisj_m_of_ising_model(model: IsingModel2D, \
                                         equilibration_iteration_count = 2**9,\
                                         sampling_count = 2**13):
    nt = 1
    T = np.linspace(1.53, 3.28, nt);
    accumulated_si_observation = np.zeros((neuron_count))
    accumulated_sisj_observation = np.zeros((neuron_count, neuron_count))

    # for each temperature points
    iT = 1.0/T[0]
    log_interval = equilibration_iteration_count // 100

    for i in range(equilibration_iteration_count):         # equilibrate
        print("equilibration: %d/%d" % (i, equilibration_iteration_count))
        if i % log_interval == 0:
            log(log_file, "equilibration: %d/%d" % (i, equilibration_iteration_count))

        mcmove(model, iT)           # Monte Carlo moves

    print("Finish Equilibration")
    log(log_file, "Finish Equilibration")
    log_interval = sampling_count // 100
    for i in range(sampling_count):
        print("sampling: %d/%d" % (i, sampling_count))
        if i % log_interval == 0:
            log(log_file, "sampling: %d/%d" % (i, sampling_count))

        mcmove(model, iT)
        accumulated_si_observation += model.configuration
        tiled_matrix = np.tile(model.configuration, neuron_count).reshape((neuron_count, neuron_count))
        accumulated_sisj_observation = ((tiled_matrix) + tiled_matrix.T) // 2
        del tiled_matrix


    si = accumulated_si_observation / (sampling_count)
    sisj = accumulated_sisj_observation / (sampling_count)
    return si, sisj

def generate_configuration_samples(model, sample_count = 1e4, equilibration_iteration_count = 1e8):
    # for each temperature points
    configuration_samples = np.zeros((sample_count, neuron_count))
    iT=1.0;
    for i in range(equilibration_iteration_count):         # equilibrate
        mcmove(model, iT)           # Monte Carlo moves

    for i in range(sample_count):
        print("generate %d-th configuration sample" % (i + 1))
        sampled_configuration = mcmove(model, iT)
        configuration_samples[i] = sampled_configuration
    return configuration_samples

def log(logfile, append_content):
	logfile.write("%s\n" % append_content)
	logfile.flush()


# Definition of data source configuration
base_file_path = os.path.join("./spike_data")
population_names = ["neo_exe_cells", "neo_inh_cells"]
dt_now = datetime.datetime.now()
time_now_string_description = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
log_file_name = "%s.log" % time_now_string_description
log_file = open(log_file_name, 'w')

spikes_record_dfs = []
for population_name in population_names:
    packet_record_file_name = "packets_%s" % population_name
    v_record_file_name = "v_%s" % population_name
    spikes_record_file_name = "spikes_%s" % population_name
    
    packet_record_file_path = os.path.join(base_file_path, packet_record_file_name + ".csv")
    v_record_file_path = os.path.join(base_file_path, v_record_file_name + ".csv")
    spikes_record_file_name = os.path.join(base_file_path, spikes_record_file_name + ".csv")
    
    # Read Records
    packet_record_df = pd.read_csv(packet_record_file_path) # Columns = slice_id?, rows = time_step
    v_record_df = pd.read_csv(v_record_file_path, header=None) # Columns = neuro_index, rows = time_step
    spikes_record_df = pd.read_csv(spikes_record_file_name, names=["neurons", "spikes"])

    spikes_record_dfs.append(spikes_record_df)
    time_steps = v_record_df.shape[0]
    neuron_count = v_record_df.shape[1]
    if v_record_df.shape[0] != packet_record_df.shape[0]:
        raise ValueError


# merge
neuron_count = 0
for spikes_record_df_tmp in spikes_record_dfs:
    current_df_neuron_count = max(spikes_record_df_tmp['neurons']) + 1
    spikes_record_df_tmp['neurons'] += neuron_count
    neuron_count += (int)(current_df_neuron_count)

spikes_record_df = pd.concat(spikes_record_dfs, axis = 0)

average_each_site_spikes_amount_in_simulation = [(neuron_index,df.size/time_steps) for neuron_index, df in list(spikes_record_df.groupby('neurons'))]

# calculate <s_i>
ensemble_avgerage_neuron_activation = np.zeros((neuron_count))
for (neuron_id, activation_ratio) in average_each_site_spikes_amount_in_simulation:
    ensemble_avgerage_neuron_activation[(int)(neuron_id)] = activation_ratio
    
# calculate <s_is_j>
ensemble_avgerage_joint_activation = np.zeros((neuron_count, neuron_count))
for i in range(time_steps):
    i_time_step_spikes = np.array(spikes_record_df[spikes_record_df.spikes == i].neurons).astype(int)
    for j in range(0, len(i_time_step_spikes)):
        for k in range(j + 1, len(i_time_step_spikes)):
            ensemble_avgerage_joint_activation[i_time_step_spikes[j], i_time_step_spikes[k]] += 1#  joint_activation[i_time_step_spikes[j][0], i_time_step_spikes[k][0]] + 1
ensemble_avgerage_joint_activation /= time_steps

import tqdm
import multiprocess
costs = []

def calculate_configuration_cost_with_fixed_size(fixed_size, H, J, samples):
    import numpy as np
    def create_configuration(fixed_size, neuron_count):
        config = []
        current_sum = 0
        while current_sum < neuron_count:
            if current_sum < 1200 and current_sum + fixed_size  > 1200:
                config.append(1200 - current_sum)
                current_sum += 1200 - current_sum
            elif current_sum >= 1200 and current_sum < 1500 and current_sum + fixed_size > 1500:
                config.append(1500 - current_sum)
                current_sum += 1500 - current_sum
            else:
                config.append(fixed_size)
                current_sum += fixed_size
        if sum(config) != neuron_count:
            raise ValueError
        return config

    def cost(config, H, J, samples, p=0.2):
        if samples.shape[1] != len(H) or len(J.shape) < 2 or samples.shape[1] != J.shape[0] or samples.shape[1] != J.shape[1]:
            raise ValueError
        sample_count = samples.shape[0]
        neuron_count = samples.shape[1]
        cost_sum = 0

        loc_map = [0] * neuron_count

        pos = 0
        core_atoms_count_record = dict({})

        for index, slice_length in enumerate(config):
            core_atoms_count_record[index] = slice_length
            for inner_slice_index in range(slice_length):
                loc_map[pos] = index
                pos += 1

        def no_normalize_p(sample):
            # sample_nor = (sample - (-1)) / 2
            extra_field_factor = -np.sum(H * sample)
            interaction_factor = 0
            neuro_count = sample.shape[0]
            tmp_2d_mat_for_calculate_interaction_factor = np.zeros((neuron_count, neuron_count))
            for i in range(neuron_count):
                tmp_2d_mat_for_calculate_interaction_factor[i] = sample[i] * sample
            interaction_factor = -np.sum(J * tmp_2d_mat_for_calculate_interaction_factor)
            p =  extra_field_factor + interaction_factor

            return p


        def loc(neuro_index):
            return loc_map[neuro_index]

        ps = []
        for i in range(sample_count):
            ps.append(no_normalize_p(samples[i]))


        nor_ps = np.exp((ps - min(ps)) / (max(ps) - min(ps) + 1))
        for i in range(sample_count):
            p = nor_ps[i]
            for j in range(0, neuron_count):
                if samples[i][j] == -1:
                    core_id_global = loc(j)
                    core_id_inner_chip = core_id_global % 18
                    chip_id = core_id_global // 18
                    atoms_inner_chip =  core_atoms_count_record[core_id_global]
                    c = 100
                    cost_sum += p * (atoms_inner_chip * p + c * (len(config) - 1))

        return cost_sum / sample_count
    return cost(create_configuration(fixed_size + 1, 1500), H, J, samples)


samples = np.load("model_samples/2024-04-03-20-22-57.npy")[-500:]


PROCESSES = 128
with multiprocess.Pool(PROCESSES) as pool:
    results = [pool.apply_async(calculate_configuration_cost_with_fixed_size, fixed_size) for fixed_size in map(lambda x: [x, H, J, samples], range(0, 1200))]
    costs = [result.get() for result in results]

import pandas as pd
print(pd.DataFrame({'d1': d1, 'd2':d2, 'd3':d3, 'd4':d4, 'costs': costs}).corr())

