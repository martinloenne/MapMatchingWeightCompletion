import sys
from app import app
import json
import datetime


# Tests the root endpoint
# Status code 200
# Correct data returned
def test_hello():
    response = app.test_client().get('/')

    assert response.status_code == 200
    assert response.data == b'The weight completion service is running! :o)'


# Tests if the info endpoint returns status code 200
def test_info_status200():
    response = app.test_client().get('/info')

    assert response.status_code == 200


# Tests if the info endpoint returns json format
def test_info_json_format():
    response = app.test_client().get('/info')

    assert response.content_type == 'application/json'


# Tests the info endpoint returns correct data in json format
def test_info_data_returned():
    response = app.test_client().get('/info')
    response_json = response.get_json()
    expected = json.loads(
        json.dumps({
            'id': 'sw502_weight_completion',
            'name': 'Weight Completion',
            'version': '2020v1',
            'category': 1
        }))

    assert response_json == expected


def test_render_good_input():
    data = {
        'timeInterval 2016-11-01': '01:15:00 - 01:30:00',
        'date': '2016-11-01'
    }
    response = app.test_client().post('/render', data=data)

    actual = response.get_json()
    readme_json = json.loads(
        json.dumps({
            'chart_type': 'markdown',
            'content': 'Weights get completed here, son'
        }))

    assert response.status_code == 200
    assert readme_json != actual


def test_render_no_input():
    response = app.test_client().post('/render')
    actual = response.get_json()
    expected = json.loads(
        json.dumps({
            'chart_type': 'markdown',
            'content': 'Weights get completed here, son'
        }))

    assert response.status_code == 200
    assert expected == actual