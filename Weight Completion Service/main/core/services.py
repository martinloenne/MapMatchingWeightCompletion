import os
import pickle
from main.common.logger import log
from main.core.weight_models import *


# Read pickles generated from weight completion algo.
class PickleLoader:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd()))

    def load_input_matrices(self):
        with open(os.path.join(self.__location__, 'resources/result_dict.pickle'), 'rb') as pickle_file:
            return pickle.load(pickle_file)['ground_truth']

    def load_prediction_matrices(self):
        with open(os.path.join(self.__location__, 'resources/result_dict.pickle'), 'rb') as pickle_file:
            return pickle.load(pickle_file)['prediction']

    def load_edges(self):
        with open(os.path.join(self.__location__, 'resources/edges.pickle'), 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def load_time_intervals(self):
        with open(os.path.join(self.__location__,
                               'resources/time_intervals_dict.pickle'), 'rb') as pickle_file:
            return pickle.load(pickle_file)['time']


class WeightCompleter:

    def __init__(self):
        # Public
        self.avail_time_intervals = None
        ############
        # Private att.
        self._limit = None
        # From pickles
        self._time_intervals_srs = None
        self._edges_lst = None
        self._input_matrices = None
        self._prediction_matrices = None

    # Load all the pickles!
    def initialize(self, pickle_loader=None):
        # Instantiate a pickle loader if none provided.
        if not pickle_loader:
            pickle_loader = PickleLoader()

        self._load_data(pickle_loader)
        self._load_avail_time_intervals()

    def _load_data(self, pickle_loader):
        self._edges_lst = pickle_loader.load_edges()
        self._input_matrices = pickle_loader.load_input_matrices()
        self._prediction_matrices = pickle_loader.load_prediction_matrices()
        self._time_intervals_srs = pickle_loader.load_time_intervals()

        # Set global index limit to num of indices in prediction results.
        self._limit = len(self._prediction_matrices)

        log("Pickles loaded.")

    # Validate retrieved from pickle files.
    def _ensure_valid_data(self):
        edge_len = len(self._edges_lst)
        input_matrix_len = len(self._input_matrices[0])
        prediction_matrix_len = len(self._prediction_matrices[0])

        if(edge_len != input_matrix_len and edge_len != prediction_matrix_len):
            raise ("Number of edges not equal to number of matrix entries")

    def _load_avail_time_intervals(self):
        time_intervals_dt = []
        for i, time_interval in enumerate(self._time_intervals_srs):
            if(i >= self._limit):
                break
            dt_object = time_interval.to_datetime()
            time_intervals_dt.append(dt_object)

        self.avail_time_intervals = time_intervals_dt


    def get_incomplete_edges(self, time_interval_start):
        return self._generate_weighted_edges(self._input_matrices, time_interval_start)


    def get_completed_edges(self, time_interval_start):
        return self._generate_weighted_edges(self._prediction_matrices, time_interval_start)


    def _generate_weighted_edges(self, weight_matrices, time_interval_start):
        log("Generating weighted edges...")
        if time_interval_start not in self.avail_time_intervals:
            raise Exception("The given time interval is not valid. Did you check available time intervals?")

        # Get index of given time interval.
        time_idx = None
        for i, avail_time in enumerate(self.avail_time_intervals):
            if avail_time == time_interval_start:
                log("Found requested time interval")
                time_idx = i
                break

        weight_matrix = weight_matrices[time_idx]

        result = WeightResult()
        result.time_interval_start = time_interval_start
        result.speed_intervals = [0, 5, 10, 15, 20, 25, 30, 35, 40]

        weighted_edges = []
        for i, edge_id in enumerate(self._edges_lst):
            w_edge = WeightedEdge()
            # Map edge.
            w_edge.edge_id = edge_id
            node_ids = edge_id.split('-')
            w_edge.start_node_osm_id  = node_ids[0]
            w_edge.end_node_osm_id  = node_ids[1]
            # Get weights from this edge in input matrix!
            w_edge.weights = weight_matrix[i].tolist()
            weighted_edges.append(w_edge)
        result.weighted_edges = weighted_edges
        return result