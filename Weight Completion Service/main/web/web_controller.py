import os
import random
import xml.etree.cElementTree as ET
from datetime import datetime, timedelta
from dateutil import tz
from flask import jsonify
from main.core.models import *


# This class encapsulates all information that the UI needs
class EdgeRepresentation:
    def __init__(self, id_, coordinates, color, weight):
        self.id = id_
        self.coordinates = coordinates
        self.color = color
        self.weight = weight


# Stores information of a node, for when they are parsed
class Node:
    def __init__(self, node_id, long, lat):
        self.nodeID = node_id
        self.long = long
        self.lat = lat

        nodeID = 0
        long = 0
        lat = 0

    def print_all(self):
        print("NodeID: ", self.nodeID, "\nlong: ", self.long, "\n", "lat: ", self.lat)


# Static instance of the weight results
class ui_results:
    weight_results = None
    speed_intervals_formatted = None


weight_results_and_formatted = ui_results()


# Make the whole UI
def make_ui():
    # Make the edge representation
    edge_representation = make_edge_representation_from_weight_completion_output()
    # When edge representation is done make the tabs
    return multiple_tabs(edge_representation)


# This method creates presentable edges from the Weight Completion output
def make_edge_representation_from_weight_completion_output():
    edge_representation_list = []

    for edge in weight_results_and_formatted.weight_results.weighted_edges:
        start, end = find_nodes(edge.start_node_osm_id, edge.end_node_osm_id)
        coords = [[start.longitude, start.latitude], [end.longitude, end.latitude]]
        hex_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        if edge.is_incomplete():
            no_weight = []
            for i in range(len(edge.weights)):
                no_weight.append("?")
            edge_representation_list.append(EdgeRepresentation(edge.edge_id, coords, hex_color, no_weight))
        else:
            rounded_weights = [round(weight, 2) for weight in edge.weights]
            edge_representation_list.append(EdgeRepresentation(edge.edge_id, coords, hex_color, rounded_weights))
    return edge_representation_list


# formats the speed intervals as string representations for tables
def format_speed_intervals(speed_interval):
    speed_intervals = []
    for i in range(len(speed_interval) - 1):
        if i == len(speed_interval) - 1:
            speed_intervals.append("%s-%s" % (speed_interval[i - 1], speed_interval[i]))
        else:
            speed_intervals.append("%s-%s" % (speed_interval[i], speed_interval[i + 1]))
    return speed_intervals


# Creates multiple tabs in UI (map and table chart)
def multiple_tabs(edge_representation):
    return (jsonify({
        "chart_type": "composite",
        "content": [
            {
                "name": "Map",
                "chart_type": "map-geo",
                "content": {
                    "view":
                        {
                            "center": {
                                "lat": 30.6646,
                                "lon": 104.0639
                            },
                            "zoom": 12
                        },
                    "featureData":
                        {
                            "type": "FeatureCollection",
                            "features": make_features(edge_representation)  # Edges on the map
                        }
                }
            },
            make_table(edge_representation)  # Table chart
        ]
    }
    ))


# Creates the features, which are the edges which result in the road network
def make_features(edge_representation):
    features = []
    # Loop through edgeRepresentation which is a list of edge class
    for edge in edge_representation:
        # Make the JSON edge
        features.append(make_edge(edge.id, edge.coordinates, edge.color))
    return features


# Gets a edgeRepresentation and inserts information in GeoJson format
def make_edge(id_, coordinates, color):
    str_id = str(id_)
    return ({
        "type": "Feature",
        "style": {
            "fill": {
                "color": color,
                "width": 0
            },
            "stroke": {
                "color": color,
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": coordinates
            },
        "properties": {
            "name": "id: " + str_id,
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    })


# Creates the table where edges and their weights are displayed
def make_table(edge_representation):
    return ({
        "name": "Weights",
        "chart_type": "simple-table",
        "content": create_columns(edge_representation)
    })


# Creates the columns of the table
def create_columns(edge_representation):
    return ({
        "settings": create_speed_interval_settings(),
        "fields": create_speed_interval_fields(),
        "data": make_tuples(edge_representation)
    })


# Creates the settings part of the table
def create_speed_interval_settings():
    count = 1
    settings = []

    ID = ({
        "objectKey": 'id',
        "sort": 'desc',
        "columnOrder": 0
    })
    settings.append(ID)
    for speed_interval in weight_results_and_formatted.speed_intervals_formatted:
        json = ({
            "objectKey": speed_interval,
            "sort": 'enable',
            "columnOrder": count
        })
        count = count + 1
        settings.append(json)
    return settings


def create_speed_interval_fields():
    fields = []

    ID = ({
        "name": 'Id',
        "objectKey": 'id'
    })
    fields.append(ID)
    for speed_interval in weight_results_and_formatted.speed_intervals_formatted:
        json = ({
            "name": speed_interval,
            "objectKey": speed_interval
        })
        fields.append(json)
    return fields


# Makes the tuple that contains column information for the table
def make_tuples(edge_representation):
    tuples = []
    for edge in edge_representation:
        id = {"id": edge.id}
        dict = id
        for i in range(len(weight_results_and_formatted.speed_intervals_formatted)):
            dict[weight_results_and_formatted.speed_intervals_formatted[i]] = edge.weight[i]
        tuples.append(dict)
    return tuples


# Makes the fields for the UI, currently the two select menus with Intervals and Dates
def make_fields(avail_time_intervals):
    time_interval_and_dates = get_time_interval_and_dates(avail_time_intervals)

    return jsonify({
        'user_fields': [
            {
                "type": "formset-select",
                "name": "date",
                "default": 1,
                "label": "Select a date",
                "options": get_date_and_intervals(time_interval_and_dates)
            },
            {
                "type": "checkbox",
                "name": "completed",
                "label": "Completed",
                "default": "true",
                "help_text": "Select to present completed network, else the uncompleted network will be displayed"
            },

        ]
    })


# Takes a date, and adds it time intervals
def get_date_and_intervals(time_interval_and_dates):
    intervals = []
    for date_and_interval in time_interval_and_dates:
        unique_select_name = make_interval_name_from_dateinterval_obj(date_and_interval)
        json = (
            {
                "name": date_and_interval.date,
                "value": date_and_interval.date,
                "fields": [{
                    "type": "select",
                    "name": unique_select_name,
                    "label": "Select time interval",
                    "options": times(date_and_interval.interval)
                }]
            }
        )
        intervals.append(json)
    return intervals


def make_interval_name_from_dateinterval_obj(date_and_interval):
    return "timeInterval" + " " + date_and_interval.date


def make_interval_name_from_date_string(date):
    return "timeInterval" + " " + date


# Make time interval into a item in a selection
def times(time_interval):
    timeIntervals = []
    for interval in time_interval:
        timeIntervals.append({"name": interval, "value": interval})
    return timeIntervals


# Class that contains a date and its available time intervals
class TimeIntervalsOnDate:
    def __init__(self):
        self.date = ""
        self.interval = []


# Use to get dates and all time intervals for given date from Weight Completion
def get_time_interval_and_dates(avail_time_intervals):
    time_intervals_on_dates = []
    # Convert
    converted_time_intervals = convert_from_utc_to_china(avail_time_intervals)
    current_date = ""
    # Get dates
    for interval in converted_time_intervals:
        # create instance of data class
        ti_on_date = TimeIntervalsOnDate()
        # Get date
        date = interval.strftime("%Y-%m-%d")
        if date != current_date:
            ti_on_date.date = date
            time_intervals_on_dates.append(ti_on_date)
            current_date = date

    # Get timestamp
    for interval in converted_time_intervals:
        # Get date
        date = interval.strftime("%Y-%m-%d")
        # Get the class
        for time_on_date in time_intervals_on_dates:
            if time_on_date.date == date:
                start_interval = interval.strftime("%H:%M:%S")
                added_interval = interval + timedelta(minutes=15)
                end_interval = added_interval.strftime("%H:%M:%S")
                time_on_date.interval.append(start_interval + " - " + end_interval)
    return time_intervals_on_dates


# Converts time interval from China time to UTC, combines date and start time and date and end time of the interval
# Output from the Weight Completion is in UTC, so it is converted to display the dates and times in Chinese timezone
def convert_from_utc_to_china(datetime_intervals):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Asia/Shanghai')
    results = []
    for x in datetime_intervals:
        converted_date = x.replace(tzinfo=from_zone).astimezone(to_zone)
        results.append(converted_date)
    return results


# The UI displays the date and times in Chinese timezone
# This is converted to UTC to be valid input for the Weight Completion
def convert_from_china_to_utc(date_time):
    from_zone = tz.gettz('Asia/Shanghai')
    to_zone = tz.gettz('UTC')
    result = date_time.replace(tzinfo=from_zone).astimezone(to_zone)
    formatted_time_string = result.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time_datetime = datetime.strptime(formatted_time_string, "%Y-%m-%d %H:%M:%S")
    return formatted_time_datetime


# Gets the readme and returns it as JSON for display
def get_readme():
    # Open README.md, load contents, return contents as json.
    with open('README.md', 'r') as content_file:
        content = content_file.read()

    return jsonify({'chart_type': 'markdown', 'content': content})


# Calls the parser, and saves to database
def parse_osm_and_save_to_db():
    nodes = parse_osm()
    add_nodes_to_db(nodes)
    return jsonify("OSM Nodes parsed and in DB")


# Parser for OSM
def parse_osm():
    # Path to the OSM file
    osm_file_path = os.path.join('.', 'main/data/chengdu.osm')
    tree = ET.parse(osm_file_path)
    root = tree.getroot()
    nodes = parse_nodes(root)
    return nodes


# Gets all nodes in OSM
def parse_nodes(root):
    # Parse nodes
    nodes = []
    count = 0
    for node in root.iter('node'):
        count += 1
        if count % 1000 == 0:
            print("We have currently parsed this many nodes: ", count)
        _nodes = []
        # Get node ID
        nodeID = node.get('id')
        # Get Long and Lat
        lat = node.get('lat')
        lon = node.get('lon')
        # Append the class
        nodes.append(Node(nodeID, lon, lat))
    return nodes


# Add all nodes to database
def add_nodes_to_db(nodes):
    count = 0
    for nd in nodes:
        count += 1
        if count > 980000:
            db_node = OSMNode(node_id=nd.nodeID, latitude=nd.lat, longitude=nd.long)
            db.session.add(db_node)
            print("We have currently added this many nodes to the session: ", count)
    db.session.commit()


# Get start & end node /w coordinates from database
def find_nodes(start_node_id, end_node_id):
    start_nd = OSMNode.query.filter_by(node_id=start_node_id).first()
    end_nd = OSMNode.query.filter_by(node_id=end_node_id).first()
    return start_nd, end_nd


# Convert date interval to datetime type
def convert_form_data_to_datetime(date, interval):
    # Split time interval to get start interval
    start, end = interval.split(' - ', 2)
    # Get date and start interval
    date_and_interval = date + " " + start
    # Make into datetime object
    converted_time = datetime.strptime(date_and_interval, "%Y-%m-%d %H:%M:%S")
    return converted_time


# returns to render endpoint
def render(date, interval, isCompleted, completer):
    if date and interval:
        time = convert_form_data_to_datetime(date, interval)
        # Convert to UTC
        converted_utc_time = convert_from_china_to_utc(time)

        # Variable which will contain date and interval which is to be displayed
        if isCompleted == "true":
            weight_results_and_formatted.weight_results = completer.get_completed_edges(converted_utc_time)
        else:
            weight_results_and_formatted.weight_results = completer.get_incomplete_edges(converted_utc_time)

        # Make speed intervals
        weight_results_and_formatted.speed_intervals_formatted = format_speed_intervals(
            weight_results_and_formatted.weight_results.speed_intervals)

        # Make and return the UI
        return make_ui()
    else:
        # Get ReadMe if first time render
        return get_readme()
