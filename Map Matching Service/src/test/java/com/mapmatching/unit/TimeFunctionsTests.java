package com.mapmatching.unit;

import com.mapmatching.core.helpers.TimeFunctions;
import com.mapmatching.unit.helpers.MyAssert;
import org.junit.jupiter.api.Test;

import java.sql.Time;
import java.util.Date;

public class TimeFunctionsTests {

    @Test
    public void estimateTimeFromTimestamp_CallWithKnownSpeedDistanceBaseTime_ReturnsCorrectlyAddedTimestamp() {
        // Arrange
        double distance = 100;
        double speed = 20;
        Date baseTime = new Date(2020, 11, 29, 12,0,0);

        Date expectedResult = new Date(2020, 11,29,12,0,5);

        // Act
        Date actualResult = TimeFunctions.estimateTimeFromTimestamp(distance, speed, baseTime, false);

        // Assert
        MyAssert.equals(expectedResult, actualResult);
    }

    @Test
    public void estimateTimeFromTimestamp_CallWithKnownSpeedDistanceBaseTime_ReturnsCorrectlySubtractedTimestamp() {
        // Arrange
        double distance = 100;
        double speed = 20;
        Date baseTime = new Date(2020, 11, 29, 12,0,0);

        Date expectedResult = new Date(2020, 11,29,11,59,55);

        // Act
        Date actualResult = TimeFunctions.estimateTimeFromTimestamp(distance, speed, baseTime, true);

        // Assert
        MyAssert.equals(expectedResult, actualResult);
    }

    @Test
    public void calcSpeedBetweenTimestamp_CallWithKnownDistanceAndTimeStamps_ReturnsCorrectSpeed() {
        // Arrange
        double distance = 100;

        Date startTime = new Date(2020,11,29,12,0,0);
        Date endTime = new Date(2020,11,29,12,0,10);

        double expectedResult = 10;

        // Act
        double actualResult = TimeFunctions.calcSpeedBetweenTimestamps(startTime, endTime, distance);

        // Assert
        MyAssert.equals(expectedResult, actualResult);
    }
}
