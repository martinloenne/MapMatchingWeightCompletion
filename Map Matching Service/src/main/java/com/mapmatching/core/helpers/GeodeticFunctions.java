package com.mapmatching.core.helpers;

import com.mapmatching.core.models.output.Point;

public class GeodeticFunctions {

    private static final int METERS_PER_KM = 1000;
    private static final int RADIUS_EARTH_KM = 6371;

    // Calculates the distance between two points using Haversines formula.
    public static double calcDistance(Point startPoint, Point endPoint) {
        double startLat = startPoint.getLatitude();
        double endLat = endPoint.getLatitude();
        double startLong = startPoint.getLongitude();
        double endLong = endPoint.getLongitude();

        // Convert to radians
        double deltaLat = Math.toRadians((endLat - startLat));
        double deltaLong = Math.toRadians((endLong - startLong));
        startLat = Math.toRadians(startLat);
        endLat = Math.toRadians(endLat);

        // Haversines formula calculates shortest distance between two points on a sphere
        double a = calcSin(deltaLat) + Math.cos(startLat) * Math.cos(endLat) * calcSin(deltaLong);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return (RADIUS_EARTH_KM * c) * METERS_PER_KM;
    }

    protected static double calcSin(double value) {
        return Math.pow(Math.sin(value / 2), 2);
    }
}
