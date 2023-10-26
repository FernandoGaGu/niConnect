# Module that groups the necessary tools to generate a report of the results given by the
# optimisation.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import os
import matplotlib.pyplot as plt
import antco
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from .plot import addDataToEdges, plotConnectomeHTML
from .io import pprint


plt.style.use('ggplot')


def generateReport(
        path: str,
        report: antco.report.Report,
        input_nodes: list,
        adj_matrix: np.ndarray,
        ref_matrix: np.ndarray,
        tar_matrix: np.ndarray,
        accessory_node: bool,
        save_connectome: bool):
    """
    Function to generate a report in the specified directory.
    """
    # create output directory
    assert not os.path.exists(path), 'Path "%s" already exists' % path
    Path(path).mkdir(parents=True)

    # save convergence
    convergence_df = pd.DataFrame(report.values).T
    convergence_df.index.names = ['iteration']
    convergence_df = convergence_df.reset_index()
    convergence_df.to_parquet(os.path.join(path, 'convergence.parquet'))

    # save convergence (img)
    antco.graphics.convergence(report, save_plot=os.path.join(path, 'convergence.png'))
    # Save the sub-graph associated to the selected nodes
    selected_nodes = report.best_solution['solution'][1:] if accessory_node else report.best_solution['solution']
    selected_nodes_names = [input_nodes[n] for n in selected_nodes]

    # generate a report with the indices of the individuals belonging to the objective cluster and
    # the nodes associated to that cluster
    output_str = '@SCORE %.5f\n' % report.best_solution['score']
    output_str += '@NODES %s' % ','.join(selected_nodes_names)
    with open(os.path.join(path, 'nodes.txt'), 'w') as out:
        out.write(output_str)
    try:
        # create adjacency matrices
        ref_matrix = pd.DataFrame(ref_matrix * adj_matrix, columns=input_nodes, index=input_nodes)
        tar_matrix = pd.DataFrame(tar_matrix * adj_matrix, columns=input_nodes, index=input_nodes)
        adj_matrix = pd.DataFrame(adj_matrix, columns=input_nodes, index=input_nodes)

        # export correlations and adjacency matrix
        ref_matrix.loc[selected_nodes_names][selected_nodes_names].to_parquet(os.path.join(path, 'ref_matrix.parquet'))
        tar_matrix.loc[selected_nodes_names][selected_nodes_names].to_parquet(os.path.join(path, 'tar_matrix.parquet'))
        adj_matrix.loc[selected_nodes_names][selected_nodes_names].to_parquet(os.path.join(path, 'adj_matrix.parquet'))

        # export plot with correlation values
        limitval = max([
            abs(min([tar_matrix.min().min(), ref_matrix.min().min()])),
            abs(max([tar_matrix.max().max(), ref_matrix.max().max()]))
        ])

        with plt.style.context('default'):
            fig, axes = plt.subplots(1, 3, figsize=(20, 4))
            plt.subplots_adjust(
                wspace=0.8
            )
            pos = axes[0].imshow(
                ref_matrix.loc[selected_nodes_names][selected_nodes_names],
                vmin=-1 * limitval, vmax=limitval, cmap='bwr'
            )
            # add colorbar
            fig.colorbar(pos, ax=axes[0])

            pos = axes[1].imshow(
                tar_matrix.loc[selected_nodes_names][selected_nodes_names],
                vmin=-1 * limitval, vmax=limitval, cmap='bwr'
            )
            # add colorbar
            fig.colorbar(pos, ax=axes[1])

            axes[2].imshow(
                adj_matrix.loc[selected_nodes_names][selected_nodes_names],
                vmin=0, vmax=1, cmap='binary'
            )

        for ax in axes.flatten():
            ax.set_xticks(np.arange(0, len(selected_nodes_names), 1))
            ax.set_xticklabels(selected_nodes_names, rotation=90)
            ax.set_yticks(np.arange(0, len(selected_nodes_names), 1))
            ax.set_yticklabels(selected_nodes_names, rotation=0)

        # add titles
        axes[0].set_title('Reference', size=15)
        axes[1].set_title('Target', size=15)
        axes[2].set_title('Adjacency matrix', size=15)

        plt.savefig(os.path.join(path, 'correlations.png'), dpi=300, bbox_inches='tight')

        # export connectomes
        if save_connectome:
            G = nx.from_pandas_adjacency(adj_matrix.loc[selected_nodes_names][selected_nodes_names])
            G = addDataToEdges(G, ref_matrix.loc[selected_nodes_names][selected_nodes_names], data_name='reference')
            G = addDataToEdges(G, tar_matrix.loc[selected_nodes_names][selected_nodes_names], data_name='target')
            plotConnectomeHTML(
                G,
                weights=True,
                graph_data_weights='reference',
                node_size=15,
                linewidth=12,
                title='Reference group',
                save_html=os.path.join(path, 'reference_connectome.html')
            )
            plotConnectomeHTML(
                G,
                weights=True,
                graph_data_weights='target',
                node_size=15,
                linewidth=12,
                title='Target group',
                save_html=os.path.join(path, 'target_connectome.html')
            )
    except Exception as ex:
        pprint('Exception generating the report.Exception:\n\n{}'.format(ex), color='red')

    # Clear matplotlib figures
    plt.close('all')

    gc.collect()  # Garbage collector
