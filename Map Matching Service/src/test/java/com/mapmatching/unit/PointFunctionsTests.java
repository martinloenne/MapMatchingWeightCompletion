package com.mapmatching.unit;

import com.mapmatching.core.helpers.EdgeFunctions;
import com.mapmatching.core.helpers.PointFunctions;
import com.mapmatching.core.models.input.RawPoint;
import com.mapmatching.core.models.output.Point;
import com.mapmatching.unit.helpers.MyAssert;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class PointFunctionsTests {

    @Test
    public void calcDistanceBetweenPoints_CallWithListOfPoints_ReturnsCorrectDistance() throws Exception {
        // Arrange
        RawPoint point1 = new RawPoint(57.03952, 9.92432, null);
        RawPoint point2 = new RawPoint(56.98104, 9.90337, null);
        RawPoint point3 = new RawPoint(56.93426, 9.90921, null);

        List<RawPoint> points = new ArrayList<>();
        points.add(point1);
        points.add(point2);
        points.add(point3);

        double expectedResult = 11839;

        // Act
        double actualResult = PointFunctions.calcDistanceBetweenPoints(points);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.5); // 0.5 meter error margin
    }

    @Test
    public void calcSpeedBetweenPoints_CallWithListOfPoints_ReturnsCorrectSpeed() throws Exception {
        // Arrange
        Date timePoint1 = new Date(2020, 10, 25, 12, 0, 0);
        Date timePoint2 = new Date(2020, 10, 25, 12, 0, 5);
        Date timePoint3 = new Date(2020, 10, 25, 12, 0, 30);

        RawPoint point1 = new RawPoint(30.6335946, 104.055136, timePoint1);
        RawPoint point2 = new RawPoint(30.6335567, 104.0547507, timePoint2);
        RawPoint point3 = new RawPoint(30.6335378, 104.0543325, timePoint3);

        List<RawPoint> points = new ArrayList<>();
        points.add(point1);
        points.add(point2);
        points.add(point3);

        double expectedResult = 2.5;

        // Act
        double actualResult = PointFunctions.calcSpeedBetweenPoints(points);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.2);
    }
}
