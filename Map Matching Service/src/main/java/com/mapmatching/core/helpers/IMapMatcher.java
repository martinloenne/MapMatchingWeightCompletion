package com.mapmatching.core.helpers;

import com.mapmatching.core.models.output.*;
import com.mapmatching.core.models.input.*;

public interface IMapMatcher {
    Route match(RawTrip trip) throws Exception;
}