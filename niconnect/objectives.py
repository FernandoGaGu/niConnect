import numpy as np
import antco
from abc import ABCMeta, abstractmethod
from functools import partial


class BaseObjective(antco.optim.ObjectiveFunction):
    """
    Class that defines the interface that the objective functions must follow in order to be used
    by the framework. The function will be maximised.

    Parameters
    ----------
    :param callable: objective
        Objective function to be maximised. This function must return a scalar.
    :param np.ndarray: target_group
        Values associated with connectivity network connections calculated at group level for
        the target group. Input shape (n_nodes, n_nodes).
    :param np.ndarray: reference_group
        Values associated with connectivity network connections calculated at group level for
        the reference group. Input shape (n_nodes, n_nodes).
    :param np.ndarray: target_suvr
        Glucose uptake values of the individuals belonging to the target group.
        Input shape: (n_target_subjects, nodes).
    :param np.ndarray: reference_suvr
        Glucose uptake values of the individuals belonging to the reference group.
        Input shape (n_reference_subjects, nodes).
    :param np.ndarray: adj_matrix
        Matrix defining the structure of the connectivity network to be explored. Input shape
        (n_nodes, n_nodes).

    Abstract methods
    ----------------
    evaluate()

    Attributes
    ----------
    target_group: np.ndarray (nodes, nodes), dtype np.float64
    reference_group: np.ndarray (nodes, nodes), dtype np.float64
    target_suvr: np.ndarray (target, nodes), dtype np.float64
    reference_suvr: np.ndarray (reference, nodes), dtype np.float64
    adj_matrix: np.ndarray (nodes, nodes), dtype np.int8
    """
    __metaclass__ = ABCMeta

    def __init__(
            self,
            objective,
            target_group: np.ndarray,
            reference_group: np.ndarray,
            target_suvr: np.ndarray,
            reference_suvr: np.ndarray,
            adj_matrix: np.ndarray,
            **kwargs
    ):
        if kwargs['objective_function_kw'] is not None:
            objective = partial(objective, **kwargs['objective_function_kw'])

        self.objective = objective
        self.reference_group = reference_group
        self.target_group = target_group
        self.target_suvr = target_suvr
        self.reference_suvr = reference_suvr
        self.adj_matrix = adj_matrix

    def evaluate(self, ant: antco.ant.Ant) -> float:
        ant_score = self.evaluateAnt(ant)
        assert isinstance(ant_score, (int, float)), \
            'Objective function must return a numeric scalar (float, int). Returned type %s' % str(type(ant_score))
        return ant_score

    @abstractmethod
    def evaluateAnt(self, ant: antco.ant.Ant) -> float:
        """
        Method that evaluates the path travelled by a given ant in such a way that it returns a
        numerical value which will be maximised.

        Parameters
        ----------
        :param antco.ant.Ant ant:
            Ant encoding the path traversed by the graph.

        Returns
        -------
        :return float
            Score associated with the ant.

        Notes
        ------
        Since the framework is optimised for parallel execution, modifications to the internal
        state of the Ant during the execution of ObjectiveFunction instances will have no effect
        on the internal state.
        """
        raise NotImplementedError

    @abstractmethod
    def getObjectiveArgs(self):
        """
        Method that must return a dictionary with the optional values required by the objective
        function received as argument in the constructor.

        Returns
        -------
        :dict
            Dictionary where the keys correspond to the optional parameters required by the
            objective  function (i.e. those parameters other than 'nodes') and the key to the value.
        """
        raise NotImplementedError

    def getHeuristic(self) -> np.ndarray:
        """ Wrapper with shape consistency for function 'getHeuristicInformation'"""
        heuristic_matrix = self.getHeuristicInformation()
        assert isinstance(heuristic_matrix, np.ndarray),\
            'getHeuristicInformation() must return a numpy.matrix of shape (n_nodes, n_nodes)'
        assert len(heuristic_matrix.shape) == 2, \
            'getHeuristicInformation() must return a numpy.matrix of shape (n_nodes, n_nodes)'
        assert (heuristic_matrix.shape[0] == self.adj_matrix.shape[0]) and (
                heuristic_matrix.shape[1] == self.adj_matrix.shape[1]), \
            'getHeuristicInformation() must return a heuristic matrix (%r) of the same shape than ' \
            'the input adjacency matrix (%r)' % (list(heuristic_matrix.shape), list(self.adj_matrix.shape))

        return heuristic_matrix

    @abstractmethod
    def getHeuristicInformation(self) -> np.ndarray:
        """
        Method that evaluates returns a matrix to be used as heuristic information during the optimization.

        Returns
        -------
        :return np.ndarray
            Heuristic information matrix.
        """
        raise NotImplementedError


class MaxGroupDiff(BaseObjective):
    """ Class that calculates the differences between the calculated connectivity networks at the
    group level of the reference group and the target group (absolute difference). """
    def __init__(self, objective, **kwargs):
        super(MaxGroupDiff, self).__init__(objective, **kwargs)

        # pr e-compute group absolute difference using the adjacency matrix
        self._abs_group_diff = np.abs(self.reference_group - self.target_group)

    def evaluateAnt(self, ant: antco.ant.Ant) -> float:
        nodes = self.getVisitedNodes(ant)  # Get nodes visited by the ant
        return self.objective(
            nodes=np.array(nodes), adj_matrix=self.adj_matrix, diff=self._abs_group_diff)

    def getObjectiveArgs(self):
        return {'adj_matrix': self.adj_matrix, 'diff': self._abs_group_diff}

    def getHeuristicInformation(self):
        return self._abs_group_diff


def costAverageEpsilonDiff(
        nodes: np.ndarray, adj_matrix: np.ndarray, diff: np.ndarray, epsilon: float) -> float:
    """ Calculate the average weighted cost, where the weights of the node-associated terms are
    calculated by applying an exponential term based on their rank
    """
    cost = 0.0
    for node_i in nodes:
        edges_diff = diff[node_i, nodes][adj_matrix[node_i, nodes] == 1]  # extract the edge differences
        edges_rank = np.argsort(-1 * edges_diff)  # calculate rank (ascending order)
        weights = (epsilon ** edges_rank)  # calculate weights
        cost += (
                np.sum(edges_diff * weights) / np.sum(weights)
        )

    return cost


# {<function name: str>: (High-level class, callable, [optional arguments names] or None)}
AVAILABLE_OBJECTIVES = {
    # Objective functions operating at the level of group differences
    'costAverageEpsilonDiff': (MaxGroupDiff, costAverageEpsilonDiff, ['epsilon'])
}


if __name__ == '__main__':
    pass

