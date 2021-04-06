package com.mapmatching.web;

import com.graphhopper.GraphHopperConfig;
import com.graphhopper.config.Profile;
import com.graphhopper.matching.MapMatching;
import com.graphhopper.util.PMap;
import com.mapmatching.core.helpers.IMapMatcher;
import com.mapmatching.core.helpers.MapMatcher;
import com.mapmatching.core.helpers.extensions.MyGraphHopper;
import com.mapmatching.core.services.*;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.List;

@SpringBootApplication
@Configuration
public class Application {

	public static void main(String[] args) {
		SpringApplication.run(Application.class, args);
	}

	// Define Bean methods that will instantiate objects used in the Spring
	// container.
	@Bean
	public IMapMatchingService MapMatchingService() {
		return new MapMatchingService();
	}

	@Bean
	public IMapMatcher MapMatcher() {
		GraphHopperConfig graphHopperConfig = makeGraphHopperConfiguration("");
		MyGraphHopper hopper = new MyGraphHopper();
		hopper.init(graphHopperConfig);
		hopper.importOrLoad();

		String name = hopper.getProfiles().get(0).getName();
		PMap hints = new PMap().putObject("profile", name);

		MapMatching mapMatching = new MapMatching(hopper, hints);

		return new MapMatcher(hopper, mapMatching);
	}

	protected GraphHopperConfig makeGraphHopperConfiguration(String osmPath) {
		GraphHopperConfig graphHopperConfig = new GraphHopperConfig();

		// Import the map
		graphHopperConfig.putObject("graph.flag_encoders", "car");
		graphHopperConfig.putObject("datareader.file", osmPath);
		graphHopperConfig.putObject("graph.location", "graph-cache");

		List<Profile> profiles = new ArrayList<>();
		profiles.add(new Profile("profile").setVehicle("car").setWeighting("fastest"));

		graphHopperConfig.setProfiles(profiles);

		return graphHopperConfig;
	}
}