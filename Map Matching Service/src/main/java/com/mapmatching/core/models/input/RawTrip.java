package com.mapmatching.core.models.input;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import com.mapmatching.common.Logger;
import com.mapmatching.web.exceptions.DataInvalidException;

public class RawTrip {
    private List<RawPoint> rawPoints;

    public RawTrip() {
        rawPoints = new ArrayList<>();
    }

    public List<RawPoint> getRawPoints() {
        return rawPoints;
    }

    public void setRawPoints(List<RawPoint> rawPoints) {
        this.rawPoints = rawPoints;
    }

    public Date getStartTime() {
        return rawPoints.get(0).getTimestamp();
    }

    public Date getEndTime() {
        return rawPoints.get(rawPoints.size() - 1).getTimestamp();
    }

    public void validate(int tripIndex) throws DataInvalidException {
        if (getRawPoints() == null || getRawPoints().isEmpty()) {
            throw new DataInvalidException("Bad input: No point list found in request.");
        }

        Date prevTimeStamp = getRawPoints().get(0).getTimestamp();

        for (int i = 0; i < getRawPoints().size(); i++) {
            RawPoint point = getRawPoints().get(i);

            if (point == null) {
                throw new DataInvalidException(
                        String.format("Bad input: Point is null (trip [%d], point [%d]). ", tripIndex, i));
            }

            if (point.getTimestamp() == null) {
                throw new DataInvalidException(
                        String.format("Bad input: Timestamp for point is null (trip [%d], point [%d]). ", tripIndex, i));
            }

            if (point.getTimestamp().before(prevTimeStamp)) {
                throw new DataInvalidException(String.format(
                        "Bad input: Timestamp invalid for point (trip [%d], point [%d]). "
                                + "The timestamp %s must be equal to or after the previous timestamp %s",
                        tripIndex, i, Logger.GetDateTimeString(point.getTimestamp()),
                        Logger.GetDateTimeString(prevTimeStamp)));
            }

            point.validate(tripIndex, i);

            prevTimeStamp = point.getTimestamp();
        }
    }
}
