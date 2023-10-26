import os
import sys
import ast
import niconnect
import time
import antco
from antco.hybrid.greedy import GreedySearchOP
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

INPUT_CONFIGURATION_FILE = os.path.join('config', 's3_optimize_networks.ini')


if __name__ == '__main__':
    # check if input arguments have been provided
    if len(sys.argv) > 1:
        if len(sys.argv) != 2:
            raise TypeError('Only one configuration file is accepted as input')
        input_config_file = sys.argv[1]
        niconnect.io.pprint('Using user-defined configuration "%s"' % input_config_file, color='green')
    else:
        input_config_file = INPUT_CONFIGURATION_FILE   # default configuration file
        niconnect.io.pprint('Using default configuration "%s"' % input_config_file, color='green')

    # read configuration file
    config = niconnect.io.INIReader.parseFile(
        input_config_file,
        required_sections=['IN.DATA', 'OUT.DATA', 'CONFIG', 'OPTIM', 'ACO']
    )

    # read configuration parameters
    conn_matrices = config['IN.DATA']['connectivity_matrices']
    suvr_values = config['IN.DATA']['suvr_values']
    output_dir = config['OUT.DATA']['directory']
    output_dir_key = config['OUT.DATA']['dir_key']
    save_connectome = config['OUT.DATA']['save_connectome']
    key_column = config['CONFIG']['key_column']
    ref_group_keys = ast.literal_eval(config['CONFIG']['reference_group'])   # read list-like objects
    tar_group_keys = ast.literal_eval(config['CONFIG']['target_group'])      # read list-like objects
    lambdas = ast.literal_eval(config['CONFIG']['lambdas'])                  # read list-like objects
    objective_function = config['OPTIM']['objective']
    objective_kw = ast.literal_eval(config['OPTIM']['objective_kw'])         # read dict-like objects
    n_nodes_start = int(config['OPTIM']['n_nodes_start'])
    n_nodes_end = int(config['OPTIM']['n_nodes_end'])
    n_nodes_step = int(config['OPTIM']['n_nodes_step'])
    n_inits = int(config['OPTIM']['n_inits'])

    # check input parameters
    assert n_nodes_start < n_nodes_end,\
        '"n_nodes_start" (%d) cannot be greater than "n_nodes_end" (%d)' % (n_nodes_start, n_nodes_end)
    assert n_nodes_start > 1, '"n_nodes_start" cannot be less than 2'
    assert n_inits >= 1, '"n_inits" cannot be less than 1'
    assert save_connectome in ['true', 'false'], 'valid values for "save_connectome" are "true" or "false"'

    save_connectome = save_connectome == 'true'  # convert from string to bool

    # read ACO parameters
    aco_param_names = {
        'integers': [
            'iterations', 'n_ants', 'tol', 'n_jobs', 'seed', 'best_ants_bag', 'best_ants_out_of_bag',
            'apply_local_search_each'
        ],
        'floats': [
            'pheromone_init', 'alpha', 'beta', 'evaporation', 'Q', 'R', 'heuristic_min_val',
            'heuristic_max_val', 'scores_min_val', 'scores_max_val'],
        'dictionaries': [
            'pheromone_update', 'local_search',
        ],
        'bools': [
            'accessory_node', 'fixed_position', 'scale_heuristic', 'scale_scores',
            'use_max_historic',
        ]}
    aco_params = {}
    for param_type, parameters in aco_param_names.items():
        if param_type == 'integers':
            for param in parameters:
                aco_params[param] = int(config['ACO'][param])
        elif param_type == 'floats':
            for param in parameters:
                aco_params[param] = float(config['ACO'][param])
        elif param_type == 'dictionaries':
            for param in parameters:
                aco_params[param] = ast.literal_eval(config['ACO'][param])
        elif param_type == 'bools':
            for param in parameters:
                param_value = config['ACO'][param]
                assert param_value in ['true', 'false'], '[ACO] "%s" valid values are [true, false]' % param
                aco_params[param] = param_value == 'true'
        else:
            assert False, 'INTERNAL ERROR (0)'   # better prevent

    n_nodes = [n for n in range(n_nodes_start, n_nodes_end+1, n_nodes_step)]
    assert len(n_nodes) >= 1, 'INTERNAL ERROR (0)'   # not enough checking
    assert aco_params['apply_local_search_each'] >= 1, '"apply_local_search_each" cannot be less than 1'

    print()
    niconnect.io.pprint('Optimizing subnetworks for number of nodes %r' % n_nodes, color='green')
    print()

    # load optimization experiment
    input_parameters = niconnect.loader.loadGroupDifferenceExperiment(
        conn_dir=conn_matrices,
        suvr_file=suvr_values,
        ref_group_keys=ref_group_keys,
        tar_group_keys=tar_group_keys,
        key_column=key_column,
        lambdas=lambdas
    )
    input_lambdas = input_parameters['lambdas']
    input_data = input_parameters['input_arguments']
    input_nodes = input_parameters['nodes']

    assert len(input_lambdas) == len(input_data), 'INTERNAL ERROR (1)'   # better prevent
    assert len(input_lambdas) == len(input_nodes), 'INTERNAL ERROR (1b)'   # better prevent

    # load objective function
    objectiveClass = niconnect.objectives.AVAILABLE_OBJECTIVES[objective_function][0]
    objective_callable = niconnect.objectives.AVAILABLE_OBJECTIVES[objective_function][1]
    objective_input_args = niconnect.objectives.AVAILABLE_OBJECTIVES[objective_function][2]

    # check objetive function optional arguments
    if objective_input_args is not None:
        for name in objective_input_args:
            if name not in objective_kw.keys():
                assert False, 'Objective function optional argument "%s" not found in "objective_kw" keys' % name

    # create output directory to save the optimization results
    output_dir = config['OUT.DATA']['directory']
    output_dir_key = config['OUT.DATA']['dir_key']
    output_dir = os.path.join(
        output_dir, '%s_%s' % (datetime.now().strftime('%Y%m%d_%H%M%S'), output_dir_key))
    assert not os.path.exists(output_dir), '"%s" already exists' % output_dir   # better prevent
    Path(output_dir).mkdir(parents=True)

    # export input configuration to the output directory
    with open(os.path.join(output_dir, 'config.ini'), 'w') as configfile:
        config.write(configfile)

    # add enter point to indicate that is a valid objective for the next pipeline step
    niconnect.io.createEnterPoint(niconnect.io.OPTIMIZATION_ENTER_POINT, output_dir)
    print()
    niconnect.io.pprint('Created output directory "%s"' % output_dir, color='green')
    print()

    # ======== Start optimizations
    number_of_optimizations = len(input_lambdas) * len(n_nodes) * n_inits
    print('\n')
    niconnect.io.pprint('Total number of optimizations to be performed: %d' % number_of_optimizations, color='cyan')
    print('\n')

    optim_count = 1
    for max_n_nodes in n_nodes:
    
        for lambda_, lambda_data, nodes in zip(input_lambdas, input_data, input_nodes):

            # create objective instance
            objective_obj = objectiveClass(
                objective=objective_callable,
                target_group=lambda_data['target_group'],
                reference_group=lambda_data['reference_group'],
                target_suvr=lambda_data['target_suvr'],
                reference_suvr=lambda_data['reference_suvr'],
                adj_matrix=lambda_data['adj_matrix'],
                objective_function_kw=objective_kw
            )

            for init in range(n_inits):
                start_subroutine = time.time()
                print()
                niconnect.io.pprint(
                    '(%d / %d) Optimizing subnetwork for lambda %.5f and %d number of nodes (init %d)' % (
                     optim_count, number_of_optimizations, lambda_, max_n_nodes, init+1), color='purple')

                # this indicates whether to scale the objective function scores
                if aco_params['scale_scores']:
                    scale_scores_obj = antco.tools.MinMaxScaler(
                        min_val=aco_params['scores_min_val'],
                        max_val=aco_params['scores_max_val'],
                        max_historic=aco_params['use_max_historic'])
                    niconnect.io.pprint('Scaling objective function scores using %s' % scale_scores_obj, color='green')
                else:
                    niconnect.io.pprint('Not scaling objective function scores', color='yellow')
                    scale_scores_obj = None

                # create ACO object
                aco_obj = antco.ACO(
                    n_ants=aco_params['n_ants'],
                    iterations=aco_params['iterations'],
                    pheromone_update=aco_params['pheromone_update'],
                    pheromone_init=aco_params['pheromone_init'],
                    evaporation=aco_params['evaporation'],
                    fixed_position=aco_params['fixed_position'],
                    alpha=aco_params['alpha'],
                    beta=aco_params['beta'],
                    tol=aco_params['tol'],
                    Q=aco_params['Q'],
                    R=aco_params['R'],
                    n_jobs=aco_params['n_jobs'],
                    seed=aco_params['seed'] + init,
                    graph=deepcopy(objective_obj.adj_matrix),
                    heuristic=deepcopy(objective_obj.getHeuristic()),
                    objective=objective_obj,
                    graph_type='u',
                    path_limits=(max_n_nodes, max_n_nodes+1),
                    precompute_heuristic=True,  # exponential heuristic information before execution
                    scaleScores=scale_scores_obj
                )

                # apply pre-processing steps
                if aco_params['scale_heuristic']:
                    niconnect.io.pprint('Scaling heuristic information to the interval [%.3f, %.3f]' % (
                        aco_params['heuristic_min_val'], aco_params['heuristic_max_val']), color='green')
                    antco.preproc.apply(aco_obj, scale_heuristic={
                        'min_val': aco_params['heuristic_min_val'],
                        'max_val': aco_params['heuristic_max_val']})
                else:
                    niconnect.io.pprint('Not scaling heuristic information', color='yellow')

                if aco_params['accessory_node']:
                    niconnect.io.pprint('Adding accessory node...', color='green')
                    antco.preproc.apply(aco_obj, accessory_node=True)
                else:
                    niconnect.io.pprint('Not using am accessory node', color='yellow')

                # create the hybrid strategy if specified
                if aco_params['local_search'] is not None:
                    metaheuristic = GreedySearchOP(
                        antco_objective=objective_obj,
                        objective=objective_obj.objective,
                        adj_matrix=deepcopy(objective_obj.adj_matrix),
                        objective_args=objective_obj.getObjectiveArgs(),
                        **aco_params['local_search'])
                    niconnect.io.pprint('Using an inner local search %s' % metaheuristic, color='green')
                else:
                    niconnect.io.pprint('Not using am inner local search', color='yellow')
                    metaheuristic = None

                # execute the optimization
                if aco_params['best_ants_bag'] == 0:
                    niconnect.io.pprint('Applying antco.algorithm.basic', color='green')
                    report = antco.algorithm.basic(
                        aco_obj=aco_obj,
                        metaheuristic=metaheuristic,
                        apply_meta_each=aco_params['apply_local_search_each'],
                        scores_decay=None,
                        evaporation_decay=None,
                        save_pheromones=False
                    )

                else:
                    niconnect.io.pprint('Applying antco.algorithm.bagOfAnts', color='green')
                    report = antco.algorithm.bagOfAnts(
                        aco_obj=aco_obj,
                        bag_size=aco_params['best_ants_bag'],
                        out_of_bag_size=aco_params['best_ants_out_of_bag'],
                        metaheuristic=metaheuristic,
                        apply_meta_each=aco_params['apply_local_search_each'],
                        scores_decay=None,
                        evaporation_decay=None,
                        save_pheromones=False
                    )

                end_subroutine = time.time()
                niconnect.io.pprint('\n... execution time %.3f seconds' % (
                        end_subroutine - start_subroutine), color='purple')
                
                # Save the sub-graph associated to the selected nodes
                niconnect.report.generateReport(
                    path=os.path.join(
                        output_dir, 'lambda-{:.5f}_nodes-{}_init-{}'.format(lambda_, max_n_nodes, init+1)),
                    report=report,
                    input_nodes=nodes,
                    adj_matrix=lambda_data['adj_matrix'],
                    ref_matrix=lambda_data['reference_group'],
                    tar_matrix=lambda_data['target_group'],
                    accessory_node=aco_params['accessory_node'],
                    save_connectome=save_connectome
                )

                optim_count += 1
