package com.mapmatching.core.helpers;

import com.graphhopper.matching.EdgeMatch;
import com.graphhopper.matching.State;
import com.graphhopper.util.*;
import com.graphhopper.util.shapes.GHPoint3D;
import com.mapmatching.core.helpers.extensions.MyGHPoint;
import com.mapmatching.core.models.input.*;
import com.mapmatching.core.models.output.*;

import java.util.ArrayList;
import java.util.List;

public class PointFunctions {
    private static final int MILLISECONDS_PER_SECOND = 1000;

    // Converts a GraphHopper PointList to a list of Points
    public static List<Point> pointListToPoints(PointList pointList) {
        List<Point> points = new ArrayList<Point>();

        for (int i = 0; i < pointList.getSize(); i++) {
            points.add(new Point(pointList.getLat(i), pointList.getLon(i)));
        }

        return points;
    }

    // Calculates the distance between a list of points
    public static double calcDistanceBetweenPoints(List<RawPoint> points) throws Exception {
        double distance = 0;

        // In order to calculate a distance, there should be at least two points
        if(points.size() < 2) {
            throw new Exception("Expected at least two points.");
        }

        for(int i = 0; i < points.size()-1; i++) {
            distance += GeodeticFunctions.calcDistance(points.get(i), points.get(i+1));
        }
        return distance;
    }

    // Calculates the average speed between a list of points
    public static double calcSpeedBetweenPoints(List<RawPoint> points) throws Exception {
        double distance = PointFunctions.calcDistanceBetweenPoints(points);

        double timeMilliseconds = points.get(points.size()-1).getTimestamp().getTime() - points.get(0).getTimestamp().getTime();
        double timeSeconds = timeMilliseconds / MILLISECONDS_PER_SECOND;

        return distance / timeSeconds;
    }

    // Extracts snapped GHPoints from an EdgeMatch and converts them to RawPoints,
    // to be added to an Edge's list of snapped points
    public static List<RawPoint> createSnappedPointsFromEdgeMatch(EdgeMatch edgeMatch) {
        List<RawPoint> points = new ArrayList<>();
        for(State state : edgeMatch.getStates()) {
            GHPoint3D snappedPoint = state.getSnap().getSnappedPoint();
            MyGHPoint originalPoint = (MyGHPoint)state.getEntry().getPoint();
            points.add(new RawPoint(snappedPoint.lat, snappedPoint.lon, originalPoint.getTimestamp()));
        }
        return points;
    }

    public static boolean coordinatesAreSame(Point firstPoint, Point secondPoint) {
        return (firstPoint.getLatitude() == secondPoint.getLatitude()) && (firstPoint.getLongitude() == secondPoint.getLongitude());
    }
}
