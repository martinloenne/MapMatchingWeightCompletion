package com.mapmatching.unit.helpers;

import org.locationtech.jts.util.Assert;

public class MyAssert extends Assert {
    public static void isWithinThreshold(double exptectedValue, double actualValue, double threshold) {
        boolean result = (exptectedValue - threshold) <= actualValue && actualValue <= (exptectedValue + threshold);
        isTrue(result);
    }
}
