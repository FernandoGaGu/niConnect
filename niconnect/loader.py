import os
import json
import warnings
import pandas as pd
import numpy as np

from . import io


def loadGroupDifferenceExperiment(
        conn_dir: str,
        suvr_file: str,
        ref_group_keys: list,
        tar_group_keys: list,
        key_column: str,
        lambdas: list
):
    """ Function used to load the information used to perform an experiment using group-level
    connectivity matrices. """

    # check input parameters
    assert os.path.exists(conn_dir), 'Directory "%s" not found' % conn_dir
    assert os.path.isdir(conn_dir), '"%s" is not a directory' % conn_dir
    assert os.path.exists(suvr_file), 'File "%s" not found' % suvr_file
    assert os.path.isfile(suvr_file), '"%s" is not a file' % suvr_file
    assert isinstance(ref_group_keys, list)
    assert isinstance(tar_group_keys, list)
    assert isinstance(lambdas, list)
    assert len(ref_group_keys) > 0, 'ref_group_keys is empty'
    assert len(tar_group_keys) > 0, 'tar_group_keys is empty'
    assert len(lambdas) > 0, 'lambdas is empty'

    # check if the input directory is a valid directory
    assert io.CONFIGURATION_ENTER_POINT in os.listdir(conn_dir), '"%s" is not a valid directory.' % conn_dir

    # check input directory consistency
    assert os.path.exists(os.path.join(conn_dir, 'conn_stats'))
    assert os.path.isdir(os.path.join(conn_dir, 'conn_stats'))
    assert os.path.exists(os.path.join(conn_dir, 'conn_matrices'))
    assert os.path.isdir(os.path.join(conn_dir, 'conn_matrices'))

    # load SUVR values
    if suvr_file.endswith('csv'):
        warnings.warn('Detected input SUVR file format "csv", parquet files are recommended.')
        suvr = pd.read_csv(suvr_file, index_col=0)
    elif suvr_file.endswith('parquet'):
        suvr = pd.read_parquet(suvr_file)
    else:
        raise TypeError('Unrecognized file format %s' % suvr_file.split('.')[-1])

    # separate SUVR values according to the provided keys
    assert key_column in suvr.columns, 'Unable to found column "%s" in file "%s"' % (key_column, suvr_file)
    ref_suvr = suvr.loc[suvr[key_column].isin(ref_group_keys)]
    tar_suvr = suvr.loc[suvr[key_column].isin(tar_group_keys)]

    print()
    io.pprint('Number of observations in the reference group(s) %r: %d' % (
        ref_group_keys, ref_suvr.shape[0]), color='green')
    io.pprint('Number of observations in the target group(s) %r: %d' % (
        tar_group_keys, tar_suvr.shape[0]), color='green')
    print()

    io.pprint('Using lambdas: %r' % lambdas, color='green')
    print()

    # load connectivity matrix
    conn_matrices = {}
    use_all_lambdas = len(lambdas) == 1 and lambdas[0] == 'all'
    # load all connectivity matrices
    for file in os.listdir(os.path.join(conn_dir, 'conn_stats')):
        # load connectivity matrix stats
        if file.endswith('json'):
            with open(os.path.join(conn_dir, 'conn_stats', file)) as f:
                conn_stats = json.load(f)
            lambda_value = conn_stats['lambda']
            n_edges = conn_stats['n_edges']
            is_connected = conn_stats['is_connected']

            if not (use_all_lambdas or
                    np.abs((np.array(lambdas) - lambda_value)).min() < 0.000001):  # tolerance for float comparison
                io.pprint('Omitting lambda %.5f' % lambda_value, color='blue')
                continue

            # load connectivity matrix structure
            in_file = os.path.join(conn_dir, 'conn_matrices', 'binary_lambda_%.5f.parquet' % lambda_value)
            assert os.path.exists(in_file), \
                'Error loading lambda connectivity matrices for value %.5f. File %s not found' % (
                    lambda_value, in_file)
            lambda_conn = pd.read_parquet(in_file)
            n_edges_loaded = lambda_conn.sum().sum() // 2    # symmetric adjacency matrix

            # check matrix consistency with the exported stats
            assert n_edges == n_edges_loaded, \
                'Incongruent number of edges between stats (%d) and connectivity matrix (%d)' % (
                    n_edges, n_edges_loaded)

            conn_matrices[lambda_value] = lambda_conn

            io.pprint('Loaded connectivity matrix for lambda %.5f' % lambda_value, color='green')

            if is_connected:
                io.pprint('The connectivity matrix is fully connected', color='green')
            else:
                io.pprint('The connectivity matrix is NOT fully connected', color='red')
            print()

    # calculate weighted connectivity matrices for the target and reference group
    ref_group_nets = {}
    tar_group_nets = {}
    input_nodes = {}
    for lambda_, conn_matrix in conn_matrices.items():
        ref_corr_lambda = ref_suvr[conn_matrix.columns].corr().values * conn_matrix.values
        tar_corr_lambda = tar_suvr[conn_matrix.columns].corr().values * conn_matrix.values

        ref_group_nets[lambda_] = ref_corr_lambda
        tar_group_nets[lambda_] = tar_corr_lambda
        input_nodes[lambda_] = list(conn_matrix.columns)

        io.pprint(
            'Average correlation difference for lambda %.5f: %.3f' % (
                lambda_,
                np.abs(ref_corr_lambda[conn_matrix == 1] - tar_corr_lambda[conn_matrix == 1]).mean()
            ), color='green')

    # convert connectivity matrices to numpy arrays
    for lambda_ in conn_matrices.keys():
        np_conn_matrix = conn_matrices[lambda_].values.astype(np.int8)
        np.fill_diagonal(np_conn_matrix, 0)  # remove diagonal entries (better prevent)
        conn_matrices[lambda_] = np_conn_matrix

    # return formatted output for niconnect.objectives.BaseObjective
    return {
        'lambdas': [lambda_ for lambda_ in conn_matrices.keys()],
        'nodes': [input_nodes[lambda_] for lambda_ in conn_matrices.keys()],
        'input_arguments': [{
            'target_group': tar_group_nets[lambda_],
            'reference_group': ref_group_nets[lambda_],
            'target_suvr': tar_suvr,
            'reference_suvr': ref_suvr,
            'adj_matrix': conn_matrices[lambda_],
        } for lambda_ in conn_matrices.keys()]
    }
