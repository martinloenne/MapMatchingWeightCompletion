package com.mapmatching.core.helpers.extensions;

import com.graphhopper.util.shapes.GHPoint;
import java.util.Date;

// An extension of the GraphHopper GHPoint containing a timestamp
public class MyGHPoint extends GHPoint {

    private Date timestamp;

    public MyGHPoint(double latitude, double longitude, Date timestamp) {
        super(latitude, longitude);
        setTimestamp(timestamp);
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }
}
