package com.mapmatching.core.helpers;

import com.graphhopper.matching.EdgeMatch;
import com.graphhopper.matching.MatchResult;
import com.graphhopper.util.EdgeIteratorState;
import com.graphhopper.util.FetchMode;
import com.graphhopper.util.PointList;
import com.mapmatching.core.models.output.*;
import com.mapmatching.core.models.input.*;
import com.mapmatching.core.helpers.extensions.MyGraphHopper;

import java.util.ArrayList;
import java.util.List;

public class EdgeFunctions {

    // Generates a list of edges from a GraphHopper MatchResult
    public static List<Edge> createEdgesFromMatchResult(MyGraphHopper hopper, MatchResult mr) {
        List<EdgeMatch> edgeMatches = mr.getEdgeMatches();
        List<Edge> edges = new ArrayList<Edge>();

        for (EdgeMatch edgeMatch : edgeMatches) {
            EdgeIteratorState edgeState = edgeMatch.getEdgeState();
            int edgeId = edgeState.getEdge();

            Edge edge = new Edge();
            edge.setOsmWayId(hopper.getOSMWay(edgeId));
            edge.setDistance(edgeState.getDistance());

            Node startNode = createStartNode(hopper, edgeState);
            Node endNode = createEndNode(hopper, edgeState);

            edge.setStartNode(startNode);
            edge.setEndNode(endNode);

            edge.setSnappedPoints(PointFunctions.createSnappedPointsFromEdgeMatch(edgeMatch));

            edges.add(edge);
        }

        return edges;
    }

    // Gets a continuous sequence of empty edges in a list of edges starting from a given index
    public static List<Edge> getSequenceEmptyEdges(List<Edge> edges, int index) {
        List<Edge> emptyEdges = new ArrayList<>();

        boolean lastEmptyEdgeFound = false;

        while(!lastEmptyEdgeFound && index < edges.size()) {
            if(edges.get(index).getSnappedPoints().size() == 0) {
                emptyEdges.add(edges.get(index));
                index++;
            }
            else {
                lastEmptyEdgeFound = true;
            }
        }
        return emptyEdges;
    }

    // Calculates the total distance between a starting edge, a list of empty edges and an ending edge
    public static double calcDistanceBetweenEmptyEdges(Edge startingEdge, Edge endingEdge, List<Edge> emptyEdges) {
        RawPoint firstPoint = startingEdge.getLastSnappedPoint(); // The starting point is the last point on starting edge
        RawPoint lastPoint = endingEdge.getFirstSnappedPoint(); // The ending point is the first point on ending edge

        double distance = GeodeticFunctions.calcDistance(firstPoint, emptyEdges.get(0).getStartNode());

        for (Edge edge : emptyEdges) {
            distance += edge.getDistance();
        }

        distance += GeodeticFunctions.calcDistance(lastPoint, emptyEdges.get(emptyEdges.size()-1).getEndNode());

        return distance;
    }

    // Calculates the total average speed between a starting edge, a list of empty edges and an ending edge
    public static double calcSpeedBetweenEmptyEdges(Edge startingEdge, Edge endingEdge, List<Edge> emptyEdges) {
        RawPoint firstPoint = startingEdge.getLastSnappedPoint();
        RawPoint lastPoint = endingEdge.getFirstSnappedPoint();

        double distance = calcDistanceBetweenEmptyEdges(startingEdge, endingEdge, emptyEdges);

        return TimeFunctions.calcSpeedBetweenTimestamps(firstPoint.getTimestamp(), lastPoint.getTimestamp(), distance);
    }

    // Calculates the average speed of an edge based on its points
    public static double calcSpeedFromEdgePoints(Edge edge) throws Exception {
        // An edge needs to have at least 2 points to calculate the average speed
        if(edge.getSnappedPoints().size() < 2) {
            throw new Exception("Edge has one or zero points.");
        }

        return PointFunctions.calcSpeedBetweenPoints(edge.getSnappedPoints());
    }

    // Calculates the average speed of an edge based on a single point on the edge
    public static double calcSpeedFromSingleEdgePoint(Edge edge, List<Edge> edges) {
        double speed;
        List<Edge> emptyEdges = getSequenceEmptyEdges(edges, edges.indexOf(edge)+1);
        // If the next edge is empty, empty edges need to be included in the calculation
        if(emptyEdges.size() > 0) {
            Edge nextEdge = edges.get(edges.indexOf(emptyEdges.get(emptyEdges.size()-1))+1);

            speed = EdgeFunctions.calcSpeedBetweenEmptyEdges(edge, nextEdge, emptyEdges);
        }
        else {
            Edge nextEdge = edges.get(edges.indexOf(edge)+1);

            speed = calcSpeedBetweenOverlappingEdges(edge, nextEdge, edge.getEndNode());

        }
        return speed;
    }

    // Calculates the speed over an overlapping node in two edges
    public static double calcSpeedBetweenOverlappingEdges(Edge firstEdge, Edge secondEdge, Node overlappingNode) {
        RawPoint firstPoint = firstEdge.getSnappedPoints().get(firstEdge.getSnappedPoints().size()-1);
        RawPoint lastPoint = secondEdge.getSnappedPoints().get(0);

        double distance = GeodeticFunctions.calcDistance(firstPoint, overlappingNode);
        distance += GeodeticFunctions.calcDistance(overlappingNode, lastPoint);

        return TimeFunctions.calcSpeedBetweenTimestamps(firstPoint.getTimestamp(), lastPoint.getTimestamp(), distance);
    }

    // Creates a start node with OSM node ID based on a GraphHopper EdgeIteratorState
    protected static Node createStartNode(MyGraphHopper hopper, EdgeIteratorState edgeState) {
        PointList towerNodes = edgeState.fetchWayGeometry(FetchMode.TOWER_ONLY);
        Node startNode = createNode(towerNodes, 0);

        long osmNodeId = hopper.getTowerOSMNode(edgeState.getBaseNode());
        startNode.setOsmNodeId(osmNodeId);

        return startNode;
    }

    // Creates an end node with OSM node ID based on a GraphHopper EdgeIteratorState
    protected static Node createEndNode(MyGraphHopper hopper, EdgeIteratorState edgeState) {
        PointList towerNodes = edgeState.fetchWayGeometry(FetchMode.TOWER_ONLY);
        Node startNode = createNode(towerNodes, 1);

        long osmNodeId = hopper.getTowerOSMNode(edgeState.getAdjNode());
        startNode.setOsmNodeId(osmNodeId);

        return startNode;
    }

    // Creates a node based on a GraphHopper PointList
    protected static Node createNode(PointList towerNodes, int index) {
        double latitude = towerNodes.getLatitude(index);
        double longitude = towerNodes.getLongitude(index);

        return new Node(latitude, longitude);
    }
}
