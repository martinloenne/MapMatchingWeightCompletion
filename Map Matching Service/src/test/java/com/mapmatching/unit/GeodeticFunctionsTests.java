package com.mapmatching.unit;

import com.mapmatching.core.helpers.GeodeticFunctions;
import com.mapmatching.core.models.output.*;
import com.mapmatching.unit.helpers.MyAssert;
import org.junit.jupiter.api.Test;

public class GeodeticFunctionsTests {

    @Test
    public void calcDistance_CallWithTwoPoints_ReturnsCorrectDistance() {
        // Arrange
        Point startPoint = new Point(57, 10);
        Point endPoint = new Point(54,9);

        double expectedResult = 339468;

        // Act
        double actualResult = GeodeticFunctions.calcDistance(startPoint, endPoint);

        // Assert
        MyAssert.isWithinThreshold(expectedResult, actualResult, 0.5); // 0.5 meter error margin
    }
}
