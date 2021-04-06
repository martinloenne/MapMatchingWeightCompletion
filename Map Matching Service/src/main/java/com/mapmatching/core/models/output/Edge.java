package com.mapmatching.core.models.output;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.graphhopper.util.EdgeIteratorState;
import com.graphhopper.util.FetchMode;
import com.graphhopper.util.PointList;
import com.mapmatching.core.helpers.extensions.MyGraphHopper;
import com.mapmatching.core.models.input.*;

import java.util.ArrayList;
import java.util.List;

public class Edge {

  private long osmWayId;
  private double averageSpeed;
  private double distance;
  private Node startNode;
  private Node endNode;
  private List<RawPoint> snappedPoints;

  public Edge() {
    snappedPoints = new ArrayList<>();
  }

  public long getOsmWayId() {
    return osmWayId;
  }

  public void setOsmWayId(long osmWayId) {
    this.osmWayId = osmWayId;
  }

  public double getAverageSpeed() {
    return averageSpeed;
  }

  public void setAverageSpeed(double averageSpeed) {
    this.averageSpeed = averageSpeed;
  }

  public double getDistance() {
    return distance;
  }

  public void setDistance(double distance) {
    this.distance = distance;
  }

  public Node getStartNode() {
    return startNode;
  }

  public void setStartNode(Node startNode) {
    this.startNode = startNode;
  }

  public Node getEndNode() {
    return endNode;
  }

  public void setEndNode(Node endNode) {
    this.endNode = endNode;
  }

  @JsonIgnore
  public List<RawPoint> getSnappedPoints() {
    return snappedPoints;
  }

  public void setSnappedPoints(List<RawPoint> snappedPoints) {
    this.snappedPoints = snappedPoints;
  }

  @JsonIgnore
  public RawPoint getFirstSnappedPoint() {
    return getSnappedPoints().get(0);
  }

  @JsonIgnore
  public RawPoint getLastSnappedPoint() {
    return getSnappedPoints().get(getSnappedPoints().size()-1);
  }
}
