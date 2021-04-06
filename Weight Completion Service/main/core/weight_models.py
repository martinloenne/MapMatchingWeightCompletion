class WeightResult:
	def __init__(self):
		self.time_interval_start = None
		self.weighted_edges = []
		self.speed_intervals = []

class WeightedEdge:
	def __init__(self):
		self.edge_id = ""
		self.start_node_osm_id = 0
		self.end_node_osm_id = 0
		self.weights = []

	def is_incomplete(self):
		return all(v == 0 for v in self.weights)
