from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Nodes(db.Model):
    __bind_key__ = 'rou'
    __tablenamme__ = 'nodes'

    nodeid = db.Column(db.BIGINT, primary_key=True)
    timestamp = db.Column(db.DATETIME, nullable=False)
    osmnodeid = db.Column(db.BIGINT, nullable=False)

    def __str__(self):
        return f"Node id: {self.nodeid}, Timestamp: {self.timestamp}, OSM node id: {self.osmnodeid}"


class Edge(db.Model):
    __bind_key__ = 'rou'
    __tablename__ = 'edges'

    edgeid = db.Column(db.BIGINT, primary_key=True)
    osmwayid = db.Column(db.FLOAT, nullable=False)
    averagespeed = db.Column(db.FLOAT, nullable=False)
    distance = db.Column(db.FLOAT, nullable=False)

    startnodenodeid = db.Column(db.BIGINT, db.ForeignKey('nodes.nodeid'))
    endnodenodeid = db.Column(db.BIGINT, db.ForeignKey('nodes.nodeid'))

    startnode = db.relationship('Nodes', foreign_keys="Edge.startnodenodeid")
    endnode = db.relationship('Nodes', foreign_keys="Edge.endnodenodeid")

    def __str__(self):
        return f"Edge id: {self.edgeid}, OSM way id: {self.osmwayid}, Average speed: {self.averagespeed}, Distance: {self.distance}, Start node id: {self.startnodenodeid}, End node id: {self.endnodenodeid}"