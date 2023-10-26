import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_aal
import networkx as nx
from copy import deepcopy
from nilearn import datasets
from nilearn import plotting


def plotConnectome(
        adj_matrix: pd.DataFrame,
        node_color: str = '#95A5A6',
        node_size: int = 20,
        edge_vmin: float = -1.05,
        edge_vmax: float = 1.05,
        edge_threshold: float = 0.05,
        title: str = None,
        colorbar: bool = True,
        dpi: int = 300,
        figsize: tuple = (15, 5),
        edge_cmap: str = 'RdYlBu_r',
        savefig: str = None
):
    """ Function used to plot a connectome based on a pandas DataFrame representing an adjacency matrix, """
    assert isinstance(adj_matrix, pd.DataFrame), 'Input adjacency matrix must be a pandas DataFrame'

    # load AAL information
    aal_atlas = fetch_atlas_aal()
    indices_labels_hash = dict(zip(aal_atlas['indices'], aal_atlas['labels']))

    # get atlas coords by roi
    atlas_coords, atlas_indices = plotting.find_parcellation_cut_coords(
                labels_img=aal_atlas.maps, return_label_names=True)
    atlas_labels = np.array([indices_labels_hash[str(idx)].lower() for idx in atlas_indices])

    # get the coords of the ROIs included in the adjacency matrix
    adj_matrix_coords = np.array([atlas_coords[atlas_labels == roi][0] for roi in adj_matrix.columns])

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)
    plotting.plot_connectome(
        adjacency_matrix=adj_matrix.values,
        node_coords=adj_matrix_coords,
        node_color=node_color,
        node_size=node_size,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_threshold=edge_threshold,
        title=title,
        axes=ax,
        colorbar=colorbar,
        edge_cmap=edge_cmap
    )
    if savefig:
        plt.savefig(savefig, dpi=dpi)
    else:
        plt.show()


def addDataToEdges(graph: nx.Graph, data: pd.DataFrame or dict, data_name: str, inplace: bool = False) -> nx.Graph:
    """ Function that allows adding information to the connections of a given network.

    Parameters
    ----------
    :param networkx.Graph graph: Network to which the information will be added.
    :param pandas.DataFrame or dict data: Data to be added to the network. This can be provided as a pandas
        DataFrame where the nodes of the network connections must be present in the index and columns
        (usually corresponding to an adjacency matrix), or in a nested dictionary.
    :param str data_name: Name of the attribute to be added to the network edges.
    :param bool inplace: Indicates whether to perform the operation inplace. Defaults to False

    :return nx.Graph: Network with updated edge information.
    """
    if not inplace:
        graph = deepcopy(graph)

    for n1, n2, metadata in graph.edges(data=True):
        if data_name in metadata:
            raise TypeError('Data "{}" already exists in the provided graph.'.format(data_name))
        if isinstance(data, pd.DataFrame):
            metadata[data_name] = data.loc[n1].loc[n2]
        else:
            if not (n1 in data or n2 in data):
                raise TypeError('Either "{}" or "{}" must be present as keys in the dictionary'.format(n1, n2))
            if not (n2 in data.get(n1, []) or n1 in data.get(n2, [])):
                raise TypeError('Either "{}" or "{}" must be keys in the nested dictionary'.format(n1, n2))
            if n1 in data:
                metadata[data_name] = data[n1][n2]
            else:
                metadata[data_name] = data[n2][n1]

    return graph


def plotConnectomeHTML(graph: nx.Graph or pd.DataFrame, weights: bool = False, graph_data_weights: str = 'weights',
                      node_color: str or dict = 'grey', default_node_color: str = 'grey', node_size: int = 2,
                      edge_cmap: str = 'bwr', edge_threshold: float = None, set_below_threshold_to: float = 0.0,
                      colorbar: bool = True, symmetric_cmap: bool = True, max_val: float = None, min_val: float = None,
                      linewidth: float = 6.0, colorbar_height: float = 0.8, colorbar_fontsize: int = 20,
                      title: str = '', title_fontsize: int = 20, edge_symmetric_threshold: float = None,
                      width: int = 600, height: int = 400, save_html: str = None, open_in_browser: bool = False):
    """ Function to represent the connectome defined by the AAL atlas.

    Parameters
    ----------
    :param nx.Graph or pandas.DataFrame graph: Graph (networkx library) whose nodes must correspond to brain
        regions associated with the AAL atlas with the nomenclature followed by nilearn.
    :param bool weights: Parameter indicating whether to represent the weights of the connections. If this parameter
        is set to True it will try to access the data associated to the connections of the graph defined by
        "graph_data_weights". Defaults to None.
    :param str graph_data_weights: Name that identifies the data associated with the weights of the network
        connections. It will only be used when parameter "weights" is True. Defaults to "weights".
    :param str or dict node_color: node color.
    :param str default_node_color: node color used by default when "node_color" is provided as dictionary.
    :param int node_size: node size.
    :param str edge_cmap: colormap used to represent the graph edges.
    :param float edge_symmetric_threshold: DESCRIPTION
    :param float edge_threshold: DESCRIPTION
    :param float set_below_threshold_to: DESCRIPTION
    :param bool colorbar: indicates whether to display the colorbar.
    :param bool symmetric_cmap: if True the colorbar will be symmetric.
    :param float min_val: DESCRIPTION
    :param float max_val: DESCRIPTION
    :param float linewidth: edges line width.
    :param float colorbar_height: colorbar height (relative to the brain image).
    :param int colorbar_fontsize: colorbar fontsize.
    :param str title: plot title.
    :param int width: plot width.
    :param int height: plot height.
    :param int title_fontsize: title font size.
    :param str save_html: parameter indicating whether to save the image as html.
    :param bool open_in_browser: parameter indicating whether to open the image in a browser.
    """
    atlas = datasets.fetch_atlas_aal()
    # atlas_coords (n_regions, 3); atlas_labels (n_regions, 1)
    atlas_coords, atlas_labels = plotting.find_parcellation_cut_coords(labels_img=atlas.maps, return_label_names=True)
    region_index_lookup = {  # {<region: str>: <atlas index: int>}
        region.lower(): int(idx) for idx, region in zip(atlas['indices'], atlas['labels'])}
    index_coords_lookup = dict(zip(atlas_labels, atlas_coords))  # {<atlas index: <int>: <atlas coords: np.array>}

    assert len(region_index_lookup) == len(
        atlas['indices']), '(1) Unitary test fails (mitools.plot.connectome.plotConnectomeAAL)'
    assert len(index_coords_lookup) == len(
        atlas['indices']), '(2) Unitary test fails (mitools.plot.connectome.plotConnectomeAAL)'

    # create the adjacency matrix
    graph_nodes = list(graph.nodes())
    adj_matrix = np.zeros(shape=(len(graph_nodes), len(graph_nodes)))   # ordered by node order in the graph
    for (node_1, node_2, data) in list(graph.edges(data=True)):
        if weights and graph_data_weights in data:
            assert graph_data_weights in data
            if edge_threshold is not None:
                if data[graph_data_weights] >= edge_threshold:
                    edge_value = data[graph_data_weights]
                else:
                    edge_value = set_below_threshold_to
            else:
                edge_value = data[graph_data_weights]
        else:
            edge_value = 1.0
        adj_matrix[graph_nodes.index(node_1), graph_nodes.index(node_2)] = edge_value
        adj_matrix[graph_nodes.index(node_2), graph_nodes.index(node_1)] = edge_value

    # get node coords (in the same order as adjacency matrix)
    node_coords = np.stack([index_coords_lookup[region_index_lookup[node]] for node in graph_nodes])

    # Hack to control the colorbar min and max values
    # Filling the diagonal does not affect the representation
    if min_val is not None:
        adj_matrix[0, 0] = min_val
    if max_val is not None:
        adj_matrix[-1, -1] = max_val

    # process node colors
    if isinstance(node_color, dict):
        node_color_ = [None] * len(graph_nodes)
        for i, node in enumerate(graph_nodes):
            node_color_[i] = node_color.get(node, default_node_color)
        node_color = node_color_

    cview = plotting.view_connectome(
        adj_matrix, node_coords, node_color=node_color, edge_threshold=edge_symmetric_threshold,
        node_size=node_size, edge_cmap=edge_cmap, symmetric_cmap=symmetric_cmap,
        linewidth=linewidth, colorbar=colorbar, colorbar_height=colorbar_height,
        colorbar_fontsize=colorbar_fontsize, title=title, title_fontsize=title_fontsize)

    cview.__dict__['width'] = width
    cview.__dict__['height'] = height

    if open_in_browser:
        cview.open_in_browser()

    if save_html:
        cview.save_as_html(save_html)

    return cview
