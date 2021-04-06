package com.mapmatching.core.models.output;

import com.mapmatching.core.models.input.RawPoint;

import java.util.Date;

public class Node extends RawPoint {
    private long osmNodeId;

    public Node(double latitude, double longitude) {
        super(latitude, longitude, null);
    }

    public long getOsmNodeId() {
        return osmNodeId;
    }

    public void setOsmNodeId(long osmNodeId) {
        this.osmNodeId = osmNodeId;
    }
}
