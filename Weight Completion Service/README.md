# Weight Completion

A Weight Completion service for the aSTEP platform made by Group SW502E20 - written in Python. 

The service uses machine learning to fill out missing weights in a road network, for a specified time interval. At the moment, the Weight Completion Service can only do weight completion in the Chinese city, **Chengdu**. 

The graph completion algorithm used in the service is based on the Github repository [GraphCompletion](https://github.com/hujilin1229/GraphCompletion), created by Jilin Hu, Chenjuan Guo, Bin Yang and Christian S. Jensen. The algorithm was created as a part of the scientific paper **"Stochastic Weight Completion for Road Networks Using Graph Convolutional Networks"**, which can be found [here](http://people.cs.aau.dk/~byang/papers/ICDE2019-GCWC.pdf).

The [Flask](https://flask.palletsprojects.com/en/1.1.x/) framework has been used as the web framework in this project.

## How to use

Please select a date from the form in the menu. After selecting a date, a specific time interval can be selected. A road network with incomplete weights can be visualized on the map by leaving the checkbox unchecked. If checked, a road network with completed weights will be shown. Two tabs are then presented. The first tab will allow the user to inspect the edges on a map, and the other will show the weights for all edges in a table.

## Running the service locally

A requirements-file has been provided in order to set up the correct dependencies. As the code uses old versions of several libraries, please ensure that these versions are correct. If you have trouble installing the requirements file, take a look on the trouble shooting comments in the requirements file.

Furtermore, a docker file has been provided if you wish to run the service inside docker. A docker file is placed in the root folder of this project. Map your local port to port 5000 of the docker container and call the endpoint at
http://localhost:YOURLOCALPORT/mapmath

## Authors

- Adil Cemalovic
- Christian Damsgaard
- Magnus Lund
- Martin Lønne
- Simon Holst
- Søren Hjorth Boelskifte
