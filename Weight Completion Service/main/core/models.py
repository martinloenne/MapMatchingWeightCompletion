from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Edge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_node_id = db.Column(db.Integer)
    end_node_id = db.Column(db.Integer)
    time_intervals = db.relationship('TimeInterval', backref='edge', lazy=True)

    def __repr__(self):
        return f"id {self.id}, start_node_id {self.start_node_id}, end_node_id {self.end_node_id}, " \
               f"time_interval {self.time_intervals}"


class SpeedDistribution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_speed = db.Column(db.Float, nullable=False)
    end_speed = db.Column(db.Float, nullable=False)
    speed_probability = db.Column(db.Float, nullable=False)
    time_interval_id = db.Column(db.Integer, db.ForeignKey('time_interval.id'), nullable=False)

    def __repr__(self):
        return f"id {self.id}, start_speed {self.start_speed}, end_speed {self.end_speed}, " \
               f"speed_probability {self.speed_probability}, time_interval_id {self.time_interval_id}"


class TimeInterval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    speed_distributions = db.relationship('SpeedDistribution', backref='time_interval', lazy=True)
    edge_id = db.Column(db.Integer, db.ForeignKey('edge.id'), nullable=False)

    def __repr__(self):
        return f"id {self.id}, start_time {self.start_time}, end_time {self.end_time}," \
               f" speed_distributions {self.speed_distributions}, edge_id {self.edge_id}"


class AdjacencyMatrix(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    row_edge_id = db.Column(db.Integer, db.ForeignKey('edge.id'), nullable=False)
    col_edge_id = db.Column(db.Integer, db.ForeignKey('edge.id'), nullable=False)
    cell_value = db.Column(db.Boolean, nullable=False)

    def __repr__(self):
        return f"id {self.id}, row_edge_id {self.row_edge_id}, col_edge_id {self.col_edge_id}," \
               f" cell_value {self.cell_value}"


association_table = db.Table('osmnodeonway', db.Model.metadata,
                            db.Column('way_id', db.BigInteger, db.ForeignKey('osmway.way_id')),
                            db.Column('node_id', db.BigInteger, db.ForeignKey('osmnode.node_id'))
                            )

class OSMWay(db.Model):
    __tablename__ = 'osmway'
    way_id = db.Column(db.BigInteger, primary_key=True)
    nodes = db.relationship(
        "OSMNode",
        secondary=association_table,
        back_populates="ways")

class OSMNode(db.Model):
    __tablename__ = 'osmnode'
    node_id = db.Column(db.BigInteger, primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    ways = db.relationship(
        "OSMWay",
        secondary=association_table,
        back_populates="nodes")

