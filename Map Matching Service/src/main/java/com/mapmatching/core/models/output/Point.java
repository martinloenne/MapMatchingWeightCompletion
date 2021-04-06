package com.mapmatching.core.models.output;

import com.mapmatching.core.helpers.GeodeticFunctions;

public class Point {
  private double latitude;
  private double longitude;

  public Point(double latitude, double longitude) {
    this.latitude = latitude;
    this.longitude = longitude;
  }

  public double getLatitude() {
    return latitude;
  }

  public void setLatitude(double latitude) {
    this.latitude = latitude;
  }

  public double getLongitude() {
    return longitude;
  }

  public void setLongitude(double longitude) {
    this.longitude = longitude;
  }
}
