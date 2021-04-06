from main.web import web_controller as web
import datetime
from dateutil import tz


def test_make_edge_():
    expected = ({
        "type": "Feature",
        "style": {
            "fill": {
                "color": "#000000",
                "width": 0
            },
            "stroke": {
                "color": "#000000",
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": "(1, 2)"
            },
        "properties": {
            "name": "id: " + "0",
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    })

    actual = web.make_edge(0, "(1, 2)", "#000000")

    assert expected == actual


def test_make_features():
    edge1 = web.EdgeRepresentation(0, '(1, 2)', "#000000", 0)
    edge2 = web.EdgeRepresentation(1, '(2, 3)', "#000001", 1)
    edge3 = web.EdgeRepresentation(2, '(3, 4)', "#000002", 2)
    edge4 = web.EdgeRepresentation(3, '(4, 5)', "#000003", 3)
    input_ = [edge1, edge2, edge3, edge4]

    x = web.make_features(input_)

    expected = ([{
        "type": "Feature",
        "style": {
            "fill": {
                "color": "#000000",
                "width": 0
            },
            "stroke": {
                "color": "#000000",
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": "(1, 2)"
            },
        "properties": {
            "name": "id: " + "0",
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    }, {
        "type": "Feature",
        "style": {
            "fill": {
                "color": "#000001",
                "width": 0
            },
            "stroke": {
                "color": "#000001",
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": "(2, 3)"
            },
        "properties": {
            "name": "id: " + "1",
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    }, {
        "type": "Feature",
        "style": {
            "fill": {
                "color": "#000002",
                "width": 0
            },
            "stroke": {
                "color": "#000002",
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": "(3, 4)"
            },
        "properties": {
            "name": "id: " + "2",
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    }, {
        "type": "Feature",
        "style": {
            "fill": {
                "color": "#000003",
                "width": 0
            },
            "stroke": {
                "color": "#000003",
                "width": 5
            }
        },
        "geometry":
            {
                "type": "LineString",
                "coordinates": "(4, 5)"
            },
        "properties": {
            "name": "id: " + "3",
            # Changes size infobox of when a edge is selected
            "size": 15
        }
    }])

    actual = web.make_features(input_)

    assert expected == actual


def test_convert_from_utc_to_china():
    input_time1 = datetime.datetime(1990, 1, 1, 23, 55, 59)
    input_time2 = datetime.datetime(1990, 1, 2, 23, 55, 59)
    input_time3 = datetime.datetime(1990, 1, 3, 23, 55, 59)
    input_ = [input_time1, input_time2, input_time3]

    expected_time1 = datetime.datetime(1990, 1, 2, 7, 55, 59, tzinfo=tz.gettz('PRC'))
    expected_time2 = datetime.datetime(1990, 1, 3, 7, 55, 59, tzinfo=tz.gettz('PRC'))
    expected_time3 = datetime.datetime(1990, 1, 4, 7, 55, 59, tzinfo=tz.gettz('PRC'))
    expected = [expected_time1, expected_time2, expected_time3]

    actual = web.convert_from_utc_to_china(input_)

    assert expected == actual


def test_convert_from_china_to_utc():
    input_ = datetime.datetime(1990, 1, 1, 23, 55, 59)

    expected = datetime.datetime(1990, 1, 1, 15, 55, 59)

    actual = web.convert_from_china_to_utc(input_)

    assert expected == actual

