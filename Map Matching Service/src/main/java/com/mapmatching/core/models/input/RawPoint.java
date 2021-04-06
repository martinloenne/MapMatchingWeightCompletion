package com.mapmatching.core.models.input;

import java.util.Date;

import com.mapmatching.core.models.output.*;
import com.mapmatching.web.exceptions.DataInvalidException;

public class RawPoint extends Point {
  private Date timestamp;

  public RawPoint(double latitude, double longitude, Date timestamp) {
    super(latitude, longitude);
    this.timestamp = timestamp;
  }

  public Date getTimestamp() {
    return timestamp;
  }

  public void setTimestamp(Date timestamp) {
    this.timestamp = timestamp;
  }

  // Throws exception to client if invalid points.
  public void validate(int tripIndex, int pointIndex) throws DataInvalidException {
    if (!(isValidLatitude(getLatitude()) && isValidLongitude(getLongitude()))) {

      throw new DataInvalidException(String.format(
              "Bad input: Point not formatted correctly (trip [%d], point [%d]). "
              + "The latitude must be between -90 and 90. The longitude must be between -180 and 180.",
              tripIndex, pointIndex));
    }
  }
  private boolean isValidLatitude(double latitude) {
    return latitude >= -90 && latitude <= 90;
  }

  private boolean isValidLongitude(double longitude) {
    return longitude >= -180 && longitude <= 180;
  }
}
