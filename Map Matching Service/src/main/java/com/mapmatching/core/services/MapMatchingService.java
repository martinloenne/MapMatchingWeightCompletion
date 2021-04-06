package com.mapmatching.core.services;

import java.util.ArrayList;
import java.util.List;

import javax.inject.Inject;

import com.mapmatching.common.*;
import com.mapmatching.core.exceptions.MapMatchingException;
import com.mapmatching.core.helpers.*;
import com.mapmatching.core.helpers.extensions.MyGraphHopper;
import com.mapmatching.core.models.input.*;
import com.mapmatching.core.models.output.*;

import jdk.jfr.Threshold;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class MapMatchingService implements IMapMatchingService {

	protected final double DISTANCE_THRESHOLD = 0.1;

	@Autowired
	protected IMapMatcher mapMatcher;

	public List<Route> mapMatch(List<RawTrip> trips) throws Exception {
		List<Route> routeResults = new ArrayList<>();

		for (RawTrip trip : trips) {
			try {
				Route route = mapMatcher.match(trip);
				routeResults.add(route);

				validateDistanceWithinThreshold(trip, route);

			// Log error and let top level handle the error.
			} catch (Exception ex) {

				if (ex.getMessage().contains("Sequence is broken")) {

					String message = ex.getMessage();
					String point = message.substring(message.indexOf("{point=") + 1, message.indexOf("}"));
					String exMessage;

					// Will contain meters if sentence is present.
					if (message.contains("distance to previous measurement?")) {
						String meters = message.substring(message.indexOf("?") + 2, message.indexOf(","));
						exMessage = "The distance between a point and adjacent points are too large (" + meters + "): "
									+ point;
					}
					// If no meters given in exception, a distance cannot be calculated.
					// The point may be outside the loaded map!
					// NOTE: This is an assumption about the exception message from the map matching
					// library and is not confirmed..
					else {
						exMessage = "Unable to map match the input. The following point may be outside the map: " + point
								+ ". Ensure that all points are within the imported map.";
					}
					Logger.Error(exMessage);
					throw new MapMatchingException(exMessage);

				}
				throw ex;
			}

		}
		return routeResults;
	}

	protected void validateDistanceWithinThreshold(RawTrip trip, Route route) throws Exception {
		double rawPointsDistance = PointFunctions.calcDistanceBetweenPoints(trip.getRawPoints());

		if(route.getDistance() > (rawPointsDistance * (1 + DISTANCE_THRESHOLD))) {
			throw new MapMatchingException("Matched result distance deviates more than "
											+ (int)(DISTANCE_THRESHOLD * 100) + "% from raw points distance.");
		}
	}
}