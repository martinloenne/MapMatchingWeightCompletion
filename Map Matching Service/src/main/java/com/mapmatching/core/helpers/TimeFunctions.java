package com.mapmatching.core.helpers;

import org.apache.commons.lang3.time.DateUtils;

import java.util.Date;

public class TimeFunctions {
    private static final int MILLISECONDS_PER_SECOND = 1000;

    // Estimates a timestamp projected from a base time, based on a given distance and speed
    public static Date estimateTimeFromTimestamp(double distance, double speed, Date baseTime, boolean subtract) {
        int milliseconds = (int) ((distance / speed) * MILLISECONDS_PER_SECOND);
        Date estimatedTime;
        // The timestamp can either be subtracted or added to the base time
        if(subtract) {
            estimatedTime = DateUtils.addMilliseconds(baseTime, -milliseconds);
        }
        else {
            estimatedTime = DateUtils.addMilliseconds(baseTime, milliseconds);
        }
        return estimatedTime;
    }

    // Calculates the average speed between two timestamp given a distance
    public static double calcSpeedBetweenTimestamps(Date startTime, Date endTime, double distance) {
        double timeMilliseconds = endTime.getTime() - startTime.getTime();
        double timeSeconds = timeMilliseconds / MILLISECONDS_PER_SECOND;

        return distance / timeSeconds;
    }
}
