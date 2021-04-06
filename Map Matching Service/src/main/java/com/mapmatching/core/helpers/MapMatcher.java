package com.mapmatching.core.helpers;

import com.graphhopper.ResponsePath;
import com.graphhopper.matching.MapMatching;
import com.graphhopper.matching.MatchResult;
import com.graphhopper.matching.Observation;
import com.graphhopper.routing.weighting.Weighting;
import com.graphhopper.util.*;
import com.mapmatching.core.helpers.extensions.MyGHPoint;
import com.mapmatching.core.models.input.*;
import com.mapmatching.core.models.output.*;
import com.mapmatching.core.helpers.extensions.MyGraphHopper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class MapMatcher implements IMapMatcher {
    protected final MyGraphHopper hopper;
    protected final PMap hints;

    protected MapMatching mapMatching;

    public MapMatcher(MyGraphHopper hopper, MapMatching mapMatching) {
        this.hopper = hopper;
        this.mapMatching = mapMatching;

        String name = hopper.getProfiles().get(0).getName();
        hints = new PMap().putObject("profile", name);
    }

    public Route match(RawTrip trip) throws Exception {
        // Measurements are created from raw trip data as input to GraphHopper's MapMatching
        List<Observation> measurements = createMeasurements(trip);

        // Raw data is map matched using GraphHopper's MapMatching
        MatchResult mr = mapMatching.match(measurements);

        // Edges and matched points are created from the match result
        List<Edge> edges = createEdges(mr);
        List<Point> matchedPoints = createMatchedPoints(mr);

        // Calculate distance and average speed for the entire route
        double routeDistance = mr.getMatchLength();

        double routeAverageSpeed = TimeFunctions.calcSpeedBetweenTimestamps(trip.getStartTime(), trip.getEndTime(), routeDistance);

        return new Route(edges, matchedPoints, routeDistance, routeAverageSpeed);
    }

    // Creates the required GraphHopper measurements from the raw trip data
    protected List<Observation> createMeasurements(RawTrip trip) {
        List<Observation> measurements = new ArrayList<Observation>();

        for (RawPoint rawPoint : trip.getRawPoints()) {
            MyGHPoint point = new MyGHPoint(rawPoint.getLatitude(), rawPoint.getLongitude(), rawPoint.getTimestamp());
            measurements.add(new Observation(point));
        }

        return measurements;
    }

    protected List<Edge> createEdges(MatchResult mr) throws Exception {
        // Create the edges without average speed and time stamps first
        List<Edge> edges = EdgeFunctions.createEdgesFromMatchResult(hopper, mr);

        if(edges.size() == 0) {
            throw new Exception("No edges in the map matched result.");
        }

        // Calculate the start time of first edge
        Edge firstEdge = edges.get(0);
        Date firstEdgeStartTime = calcFirstEdgeStartTime(edges);
        firstEdge.getStartNode().setTimestamp(firstEdgeStartTime);

        // Calculate the speed of first edge
        double firstEdgeSpeed = calcFirstEdgeSpeed(edges);
        firstEdge.setAverageSpeed(firstEdgeSpeed);

        // Loop iterates through all edges starting from the second
        // and calculates their speed based on their snapped points
        if(edges.size() > 1) {
            List<Edge> emptyEdges = new ArrayList<>(); // Buffer for empty edges
            Edge previousEdge = firstEdge; // In the first iteration, the previous edge is the first edge

            for(int i = 1; i < edges.size(); i++) {
                Edge edge = edges.get(i);
                double speed;

                // If current edge has no points, add it to the buffer and continue to next iteration
                if(edge.getSnappedPoints().size() == 0) {
                    emptyEdges.add(edge);
                }
                else {
                    // Otherwise calculate the edge's speed and apply it to the edge
                    speed = calcCurrentEdgeSpeed(edges, edge);
                    edge.setAverageSpeed(speed);

                    // If the buffer is not empty, the speed for all edges in the buffer must be calculated
                    if(emptyEdges.size() > 0) {
                        calcEmptyEdgeBufferSpeed(emptyEdges, previousEdge, edge);
                    }
                    previousEdge = edge; // The previous edge in next iteration is the current edge
                }
            }
        }
        // Finally, calculate all edge timestamps based on their speed
        // starting from the first edge start time
        calcAllEdgeTimestamps(edges, firstEdgeStartTime);

        return edges;
    }

    // Calculates the timestamps for all edges based on a given start time and their average speed
    protected void calcAllEdgeTimestamps(List<Edge> edges, Date start) throws Exception {
        Date startTime = start;
        for(Edge edge : edges) {
            if(edge.getAverageSpeed() == 0) {
                throw new Exception("Timestamp calculation was attempted without average speed.");
            }
            // The given start time is assigned to the edge's start node
            edge.getStartNode().setTimestamp(startTime);
            // The end time of the edge is calculated based on the start time and average speed
            Date endTime = TimeFunctions.estimateTimeFromTimestamp(edge.getDistance(), edge.getAverageSpeed(), startTime, false);
            // The end time is assigned to the edge and set to be
            // the start time for the next edge in the iteration
            edge.getEndNode().setTimestamp(endTime);
            startTime = endTime;
        }
    }

    // Calculates the speed for all edges in the empty edge buffer
    protected void calcEmptyEdgeBufferSpeed(List<Edge> emptyEdges, Edge previousEdge, Edge edge) {
        double emptyEdgeSpeed = EdgeFunctions.calcSpeedBetweenEmptyEdges(previousEdge, edge, emptyEdges);
        for(Edge emptyEdge : emptyEdges) {
            emptyEdge.setAverageSpeed(emptyEdgeSpeed);
        }
        emptyEdges.clear();
    }

    // Calculates the speed for the current edge
    protected double calcCurrentEdgeSpeed(List<Edge> edges, Edge edge) throws Exception {
        double speed;
        // No empty edges should be called with this method
        if(edge.getSnappedPoints().size() == 0) {
            throw new Exception("Current edge has no snapped points, but was not added to buffer.");
        }
        else if(edge.getSnappedPoints().size() > 1 && !PointFunctions.coordinatesAreSame(edge.getFirstSnappedPoint(), edge.getLastSnappedPoint())) {
            // If the edge has multiple points, calculate the speed based on its points
            speed = EdgeFunctions.calcSpeedFromEdgePoints(edge);
        }
        else {
            // Otherwise, calculate the speed based on a single point
            if(edge != edges.get(edges.size()-1)) { // Check if not the last edge, as this is a special case
                speed = EdgeFunctions.calcSpeedFromSingleEdgePoint(edge, edges);
            }
            else {
                speed = calcLastEdgeSpeed(edges); // Special method for calculating last edge speed
            }
        }
        return speed;
    }

    // Calculates the start time for the first edge
    protected Date calcFirstEdgeStartTime(List<Edge> edges) throws Exception {
        Edge firstEdge = edges.get(0);

        // Calculate the speed of the first edge
        double speed = calcFirstEdgeSpeed(edges);
        // Then calculate the distance between the start node of the first edge
        // and its first snapped point
        double distance = GeodeticFunctions.calcDistance(firstEdge.getFirstSnappedPoint(), firstEdge.getStartNode());

        // Estimate the edge start time by subtracting the appropriate time from
        // the timestamp of the first snapped point, based on the calculated speed and distance
        return TimeFunctions.estimateTimeFromTimestamp(distance, speed, firstEdge.getFirstSnappedPoint().getTimestamp(), true);
    }


    // Calculates the speed for the first edge
    protected double calcFirstEdgeSpeed(List<Edge> edges) throws Exception {
        Edge firstEdge = edges.get(0);
        double speed;

        // First edge should always have at least 1 point
        if(firstEdge.getSnappedPoints().size() == 0) {
            throw new Exception("First edge has no snapped points.");
        }
        else if(firstEdge.getSnappedPoints().size() > 1) {
            // If the first edge has multiple points, calculate the speed based on its points
            speed = PointFunctions.calcSpeedBetweenPoints(firstEdge.getSnappedPoints());
        }
        else {
            // There should always be at least 2 points if the first edge is the only edge
            if(edges.size() < 2) {
                throw new Exception("First edge has only one snapped point, but it is the only edge in the result.");
            }
            Edge secondEdge = edges.get(1);
            if(secondEdge.getSnappedPoints().size() == 0) {
                // If the second edge is empty, get all empty edges
                // in a sequence, starting from the second edge
                List<Edge> emptyEdges = EdgeFunctions.getSequenceEmptyEdges(edges, 1);

                // Get the ending edge, that is the edge directly after the last empty edge
                int lastEmptyEdgeIndex = edges.indexOf(emptyEdges.get(emptyEdges.size()-1));
                Edge endingEdge = edges.get(lastEmptyEdgeIndex+1);

                // Calculate the total average speed between the first edge, empty edges and ending edge
                speed = EdgeFunctions.calcSpeedBetweenEmptyEdges(firstEdge, endingEdge, emptyEdges);
            }
            else {
                // If the second edge has points, then the speed can be
                // calculated based on the first and second edge's points
                speed = EdgeFunctions.calcSpeedBetweenOverlappingEdges(firstEdge, secondEdge, firstEdge.getEndNode());
            }
        }
        return speed;
    }

    // Calculates the speed for the last edge
    protected double calcLastEdgeSpeed(List<Edge> edges) throws Exception {
        Edge lastEdge = edges.get(edges.size()-1);
        double speed;

        // Last edge should always have at least 1 snapped point
        if(lastEdge.getSnappedPoints().size() == 0) {
            throw new Exception("Last edge has no snapped points.");
        }
        else if(lastEdge.getSnappedPoints().size() > 1  && !PointFunctions.coordinatesAreSame(lastEdge.getFirstSnappedPoint(), lastEdge.getLastSnappedPoint())) {
            // If the last edge has multiple points, calculate the speed based on its points
            speed = PointFunctions.calcSpeedBetweenPoints(lastEdge.getSnappedPoints());
        }
        else {
            // If the second to last edge is empty, get empty edges
            Edge secondToLastEdge = edges.get(edges.size()-2);
            if(secondToLastEdge.getSnappedPoints().size() == 0) {
                // Revert the edge list, so a reverse sequence of empty edges
                // can be retrieved, starting from the second to last edge
                List<Edge> revertedEdges = new ArrayList<>(edges);
                Collections.reverse(revertedEdges);

                List<Edge> emptyEdges = EdgeFunctions.getSequenceEmptyEdges(revertedEdges, 1);

                // Get the starting edge, that is the edge before after the first empty edge
                int lastEmptyEdgeIndex = edges.indexOf(emptyEdges.get(emptyEdges.size()-1));
                Edge startingEdge = edges.get(lastEmptyEdgeIndex-1);

                // Calculate the total average speed between the starting edge, empty edges and last edge
                speed = EdgeFunctions.calcSpeedBetweenEmptyEdges(startingEdge, lastEdge, emptyEdges);
            }
            else {
                // If the second to last edge has points, then the speed can be
                // calculated based on the last and second last edge's points
                speed = EdgeFunctions.calcSpeedBetweenOverlappingEdges(secondToLastEdge, lastEdge, lastEdge.getStartNode());
            }
        }
        return speed;
    }

    // Creates the matched points from the GraphHopper match result
    protected List<Point> createMatchedPoints(MatchResult mr) {
        Translation translation = new TranslationMap().doImport().getWithFallBack(Helper.getLocale(""));

        Weighting weighting = hopper.createWeighting(hopper.getProfiles().get(0), hints);

        ResponsePath responsePath = new PathMerger(mr.getGraph(), weighting).doWork(PointList.EMPTY,
                Collections.singletonList(mr.getMergedPath()), hopper.getEncodingManager(), translation);

        PointList pointList = responsePath.getPoints();

        return PointFunctions.pointListToPoints(pointList);
    }
}
