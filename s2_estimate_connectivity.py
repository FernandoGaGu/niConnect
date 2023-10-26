# TODO. (1) Add connectome representation
import os
import sys
import niconnect
import ast
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from inverse_covariance import QuicGraphicalLasso
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

plt.style.use('ggplot')
sys.path.append(os.path.join('.', 'libs'))   # add auxiliary libraries

import mitools as mi


INPUT_CONFIGURATION_FILE = os.path.join('config', 's2_estimate_connectivity.ini')


def estimateGraphicalLassoQUIC(data: pd.DataFrame, alpha: float) -> tuple:
    """ Subroutine used to estimate connectivity matrices using the QUICLasso method implemented in
        https://github.com/skggm/skggm
     """
    model = QuicGraphicalLasso(
        lam=alpha,
        init_method='corrcoef',
        auto_scale=False)
    model = model.fit(data.values)

    # binarize the precision matrix
    precision = model.precision_
    np.fill_diagonal(precision, 0.0)
    precision[np.isclose(precision, 0, atol=0.001)] = 0
    precision[~np.isclose(precision, 0, atol=0.001)] = 1

    # estimate Pearson correlations
    correlations = np.corrcoef(data.values, rowvar=False)

    # calculate a weighted connectivity graph
    conn_graph = pd.DataFrame(precision.astype(int) * correlations, columns=data.columns, index=data.columns)
    conn_graph_bin = pd.DataFrame(precision.astype(int), columns=data.columns, index=data.columns)

    # analyze the resulting graph
    G = nx.from_pandas_adjacency(conn_graph_bin)
    graph_stats = {
        'lambda': alpha,
        'n_edges': G.number_of_edges(),
        'is_connected': nx.is_connected(G)
    }

    return conn_graph_bin, conn_graph, graph_stats


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
        required_sections=['IN.DATA', 'OUT.DATA', 'CONFIG']
    )

    # load configuration parameters
    in_file = config['IN.DATA']['file']
    in_file_format = config['IN.DATA']['format']
    out_dir = config['OUT.DATA']['directory']
    out_dir_key = config['OUT.DATA']['dir_key']
    lambdas = ast.literal_eval(config['CONFIG']['lambdas'])            # read list-like objects
    subject_keys = ast.literal_eval(config['CONFIG']['subject_keys'])  # read list-like objects
    key_column = config['CONFIG']['key_column']
    remove_cerebellum = config['CONFIG']['remove_cerebellum']

    assert remove_cerebellum in ['true', 'false'], '[CONFIG] remove_cerebellum valid values are [true, false]'

    remove_cerebellum = remove_cerebellum == 'true'

    # read input data
    assert os.path.exists(in_file), 'Input file "%s" not found' % in_file
    assert os.path.isfile(in_file), '"%s" is not a file' % in_file
    assert in_file_format in ['parquet', 'csv'], \
        'Invalid input file format "%s". Valid formats are: parquet or csv' % in_file_format

    if in_file_format == 'parquet':
        in_data = pd.read_parquet(in_file)
    elif in_file_format == 'csv':
        in_data = pd.read_csv(in_file, index_col=0)
    else:
        raise TypeError('Invalid output format "%s"' % in_file_format)
    niconnect.io.pprint('Input data shape: %r' % list(in_data.shape), color='green')

    # check input data consistency
    if key_column not in in_data.columns:
        raise TypeError('Column "%s" not found in the input data "%s"' % (key_column, in_file))

    # select subjects for estimating the connectivity matrices
    unique_data_keys = in_data[key_column].unique()
    for key in subject_keys:
        if key not in unique_data_keys:
            niconnect.io.pprint('WARNING. No entries beloging to key "%s"' % key, color='yellow')
    in_conn_data = in_data.loc[in_data[key_column].isin(subject_keys)]
    selection_info = in_conn_data[key_column].value_counts().to_dict()
    for key, value in selection_info.items():
        niconnect.io.pprint('Number of selected entries for key "%s": %d' % (key, value), color='green')

    # select aal columns
    in_conn_data_rois = in_conn_data[list(mi.roi.AALCode.alias.keys())]

    # remove cerebellum (optional)
    if remove_cerebellum:
        in_conn_data_rois = in_conn_data_rois[[
            c for c in in_conn_data_rois.columns if c not in mi.roi.ReferenceROI.aal_cerebellum]]

    # estimate connectivity matrices for the different lambda values
    print()
    connectivity_graphs = {}
    for lambda_ in tqdm(lambdas, desc='Extracting connectivity matrices...'):
        bin_am, am, stats = estimateGraphicalLassoQUIC(in_conn_data_rois, lambda_)
        connectivity_graphs[lambda_] = {
            'adjacency_matrix': am,
            'binary_adjacency_matrix': bin_am,
            'stats': stats}

    # save connectivity matrices and stats
    # ... create a valid output directory
    curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(out_dir, '%s_%s' % (curr_time, out_dir_key))
    assert not os.path.exists(output_dir), '"%s" already exists' % output_dir   # better prevent
    Path(output_dir).mkdir(parents=True)

    # add enter point to indicate that is a valid objective for the next pipeline step
    niconnect.io.createEnterPoint(niconnect.io.CONFIGURATION_ENTER_POINT, output_dir)

    # export weighted adjacency matrices (parquet format)
    Path(os.path.join(output_dir, 'conn_matrices')).mkdir(parents=True)
    Path(os.path.join(output_dir, 'conn_stats')).mkdir(parents=True)
    Path(os.path.join(output_dir, 'conn_img')).mkdir(parents=True)
    for lambda_, lambda_values in tqdm(connectivity_graphs.items(), desc='Exporting connectomes...'):
        # export conn matrix
        lambda_values['adjacency_matrix'].to_parquet(
            os.path.join(output_dir, 'conn_matrices', 'lambda_%.5f.parquet' % lambda_)
        )
        # export conn matrix
        lambda_values['binary_adjacency_matrix'].to_parquet(
            os.path.join(output_dir, 'conn_matrices', 'binary_lambda_%.5f.parquet' % lambda_)
        )
        # export graph stats
        out_json_file = os.path.join(output_dir, 'conn_stats', 'lambda_%.5f.json' % lambda_)
        with open(out_json_file, 'w') as outfile:
            json.dump(lambda_values['stats'], outfile)

        # export connectivity graphs on brains
        niconnect.plot.plotConnectome(
            lambda_values['adjacency_matrix'],
            title=r'$\lambda = {}$'.format(lambda_),
            savefig=os.path.join(output_dir, 'conn_img', 'lambda_%.5f.png' % lambda_))

    # export graph with the number of edges according lambda
    sorted_lambdas = sorted(lambdas)
    fig, ax = plt.subplots()
    ax.plot(
        sorted_lambdas,
        [connectivity_graphs[lambda_]['stats']['n_edges'] for lambda_ in sorted_lambdas]
    )
    ax.scatter(
        sorted_lambdas,
        [connectivity_graphs[lambda_]['stats']['n_edges'] for lambda_ in sorted_lambdas]
    )
    ax.set_title(r'Number of edges according $\lambda$', size=15)
    ax.set_ylabel('Number of edges', size=12)
    ax.set_xlabel('$\lambda$', size=12)
    plt.savefig(os.path.join(output_dir, 'Number_of_edges.png'), dpi=300)
    plt.close(fig)
