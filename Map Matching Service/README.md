# MapMatching

A GraphHopper based map matching service for the aSTEP platform made by group SW502E20.

The service makes use of the open source GraphHopper routing libraries:

- https://github.com/graphhopper/map-matching
- https://github.com/graphhopper/graphhopper

The project is written in Java and use the [Spring Boot](https://spring.io/projects/spring-boot) web framework.

## About the service

The map matching is perfomed based on map data from Open Street Map.

NB! Currently, only trips in the Chinese city [Chengdu](https://goo.gl/maps/G99BkaDptNdFfw5H7) can be map matched. If you wish to map match another area, fork the project and import a new area as an OSM-file by using GraphHopper's methods.

## Integration with aSTEP

This service is implemented as a REST API and can be used freely by all current and future services on the aSTEP platform.

Multiple map matching services already exists on aSTEP. However, this service offers more data as described in the section **How to use**.

The service has been made in collaboration with another semester group, Group SW505E20, whose service calls the API. Group SW505 have set up a GUI and database in order to make the map matching functionality in this service available on the aSTEP platform:

- Gitlab repository: https://daisy-git.cs.aau.dk/astep-2020-fall/sw505-test-service
- aSTEP: https://astep.cs.aau.dk/tool/astep-2020-fall-sw505-test-service.astep-dev.cs.aau.dk

## How to use

The service exposes the endpoint **/mapmatch** which accepts a POST request containing raw GPS points and returns the generated map matched points combined with other data that may be useful. Input and output data are both in JSON format.

### Expected input

A list of one or more trips containing a list of one or more raw GPS points. Every GPS point must include a latitude, longitude and timestamp. The format of the expected input can be seen here:

```
[
  {
    "rawPoints": [
      {
        "latitude": 30.62377,
        "longitude": 104.02045,
        "timestamp": "2020-10-21T11:07:41Z"
      },
      {
        "latitude": 30.62665,
        "longitude": 104.02243,
        "timestamp": "2020-10-21T11:08:43Z"
      },
      {
            ......
            ......
      }
    ]
  },
  {
    "rawPoints": [
      {
            ......
            ......
      }
    ]
  }
]
```

### Output

A list of one or more routes containing a list of edges.

An edge corresponds to a segment consisting of a start and end OSM node in the OSM map, where each node reflects a road intersection. Furthermore, an edge contains a distance, which reflects the length of the edge in meters. It also contains information about the average speed traveled on the edge in meters per second. Finally, each edge belongs to an OSM way, which is represented by a OSM way ID.

A route also has a list of matched points, which is a list of points with a latitude and longitude that are snapped to the edge. Finally, a route also has a total distance in meters, as well as the average speed traveled on the whole route. The format of the output can be seen here:

```
[
    {
      "edges": [
          {
              "osmWayId": 345684193,
              "startNode": {
                  "osmNodeId": 4972899580,
                  "timestamp": "2020-12-14T10:43:32.733+00:00",
                  "latitude": 30.633103600884308,
                  "longitude": 104.04822928566253
              },
              "endNode": {
                  "osmNodeId": 4972899582,
                  "timestamp": "2020-12-14T10:43:34.303+00:00",
                  "latitude": 30.632922924300797,
                  "longitude": 104.04861261804281
              },
              "distance": 41.818,
              "averageSpeed": 26.62481282559887
          },
          {
            ...
          },
          ...
      ],
      "matchedPoints": [
          {
              "latitude": 30.6330068834206,
              "longitude": 104.04843448621955
          },
          {
            ...
          },
          ...
      ],
      "distance": 622.9006982605781,
      "averageSpeed": 15.192699957575076
    }
]
```

Please note that data is not saved in this service. The client is responsible for saving the returned result in their own service if necessary.

## Running the service locally

Just use Docker! 🐳

A docker file is placed in the root folder of this project. Map your local port to port 5000 of the docker container and call the endpoint at
`http://localhost:YOURLOCALPORT/mapmath`

A number of files containing valid, as well as invalid (for testing purposes), input in JSON are placed in the /files folder and can be used to test the service.

## Authors

- Adil Cemalovic
- Christian Damsgaard
- Magnus Lund
- Martin Lønne
- Simon Holst
- Søren Hjorth Boelskifte
