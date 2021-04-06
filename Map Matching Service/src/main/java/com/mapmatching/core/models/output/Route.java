package com.mapmatching.core.models.output;

import java.util.List;

public class Route {

  private double averageSpeed;
  private double distance;
  private List<Edge> edges;
  private List<Point> matchedPoints;

  public Route(List<Edge> edges, List<Point> matchedPoints, double distance, double averageSpeed) {
    this.edges = edges;
    this.matchedPoints = matchedPoints;
    this.distance = distance;
    this.averageSpeed = averageSpeed;
  }

  public List<Edge> getEdges() {
    return edges;
  }

  public void setEdges(List<Edge> edges) {
    this.edges = edges;
  }

  public double getDistance() {
    return distance;
  }

  public void setDistance(double distance) {
    this.distance = distance;
  }

  public double getAverageSpeed() {
    return averageSpeed;
  }

  public void setAverageSpeed(double averageSpeed) {
    this.averageSpeed = averageSpeed;
  }

  public List<Point> getMatchedPoints() {
    return matchedPoints;
  }

  public void setMatchedPoints(List<Point> matchedPoints) {
    this.matchedPoints = matchedPoints;
  }
}
