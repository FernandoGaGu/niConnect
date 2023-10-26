# Module that groups the necessary tools for the creation and incorporation of hybrid search
# strategies within the framework.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from antco import hybrid as antco_h
from .objectives import BaseObjective


def get_hybrid(name: str, objective: BaseObjective, adj_matrix: np.ndarray, parameters: dict):
    """
    Function to load a hybrid optimisation strategy from the name and arguments received as
    parameters.

    Parameters
    ----------
    name: str
        Name of the search strategy. Currently implemented:

            - 'GreedySearchOP'

        To see available hybrid strategies use:

            help(niconnect.hybrid.AVAILABLE_HYBRID_STRATEGIES.keys())

    objective: niconnect.objectives.BaseObjective subclass
        Objective function used in the optimisation.

    adj_matrix: np.ndarray (nodes, nodes)
        Adjacency matrix defining the graph structure to be explored.

    parameters: dict
        Parameters required by the hybrid strategy. To view the required parameters use:

            help(niconnect.hybrid.AVAILABLE_HYBRID_STRATEGIES["name_of_the_strategy"])

    Returns
    -------
    :antco.hybrid.base.MetaHeuristic subclass.
        Hybrid strategy.

    Notes
    -----
    The 'objective' and 'adj_matrix' parameters of the hybrid strategy required are automatically
    provided by the framework.
    """
    assert name in AVAILABLE_HYBRID_STRATEGIES, 'Unrecognised hybrid strategy %s' % name
    assert isinstance(objective, BaseObjective), \
        'objective must be a subclass of niconnect.objective.BaseObjective. Provided: %s' \
        % type(objective)

    hybrid_strategy = AVAILABLE_HYBRID_STRATEGIES[name](
        antco_objective=objective,
        objective=objective.objective,
        adj_matrix=adj_matrix,
        objective_args=objective.get_objective_args(),
        **parameters)

    return hybrid_strategy


AVAILABLE_HYBRID_STRATEGIES = {
    # Greedy search strategy incorporated in the ACO algorithm to improve the intermediate
    # solutions applying a local search
    'GreedySearchOP': antco_h.greedy.GreedySearchOP,
}
