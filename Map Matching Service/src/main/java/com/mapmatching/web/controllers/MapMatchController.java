package com.mapmatching.web.controllers;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StopWatch;

import java.util.List;

import javax.ws.rs.InternalServerErrorException;

import com.mapmatching.common.*;
import com.mapmatching.core.exceptions.*;
import com.mapmatching.core.models.input.RawTrip;
import com.mapmatching.core.models.output.Route;
import com.mapmatching.core.services.IMapMatchingService;
import com.mapmatching.web.exceptions.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@CrossOrigin
@RestController
public class MapMatchController {

  @Autowired
  private IMapMatchingService mapMatchingService;

  // For health checking by Kubernetes.
  @RequestMapping("/")
  public ResponseEntity<String> healthCheck() {
    return new ResponseEntity<>("OK", HttpStatus.OK);
  }

  @PostMapping("/mapmatch")
  public ResponseEntity<List<Route>> PostMapMatch(@RequestBody List<RawTrip> rawTrips) {
    Logger.Info("Serving new request at POST /mapmatch");

    StopWatch stopWatch = new StopWatch();
    stopWatch.start();

    try {
      EnsureValidTrips(rawTrips);
    } catch (DataInvalidException ex) {
      Logger.Error(ex.getMessage());
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, ex.getMessage());
    } catch (Exception ex) {
      Logger.Error("An unexpected error occured while validating request:");
      ex.printStackTrace();
      throw new InternalServerErrorException(
          "An internal error occured while validating your input.");
    }

    stopWatch.stop();
    Logger.Info("Validation took " + stopWatch.getTotalTimeMillis() + "ms.");

    // Perform the map matching!
    stopWatch.start();
    List<Route> matchedRoutes;
    try {
      matchedRoutes = mapMatchingService.mapMatch(rawTrips);
      // Catch map match exception!
    } catch (MapMatchingException ex) {
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, ex.getMessage());
    } catch (Exception ex) {
      Logger.Error("An unexpected error occurred while map matching request:" + ex.getMessage());
      ex.printStackTrace();
      throw new InternalServerErrorException(
          "An internal error occurred while map matching your input.");

    }
    stopWatch.stop();
    Logger.Info("Map matching took " + stopWatch.getTotalTimeMillis() + "ms.");

    return new ResponseEntity<>(matchedRoutes, HttpStatus.OK);
  }

  // Error handling. If any errors, this method will throw
  // a DataInvalidException with a user friendly message for the client.
  private void EnsureValidTrips(List<RawTrip> rawTrips) throws DataInvalidException {
    Logger.Info("Validating " + rawTrips.size() + " trips.");

    if (rawTrips == null || rawTrips.isEmpty()) {
      throw new DataInvalidException("Bad input: JSON data is null or empty");
    }
    for (int i = 0; i < rawTrips.size(); i++) {
      RawTrip trip = rawTrips.get(i);
      trip.validate(i);
    }
  }
}