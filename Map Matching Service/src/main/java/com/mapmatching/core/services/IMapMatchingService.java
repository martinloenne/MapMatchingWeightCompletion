package com.mapmatching.core.services;

import java.util.List;

import com.mapmatching.core.models.input.RawTrip;
import com.mapmatching.core.models.output.Route;

public interface IMapMatchingService {
  List<Route> mapMatch(List<RawTrip> trips) throws Exception;
}
