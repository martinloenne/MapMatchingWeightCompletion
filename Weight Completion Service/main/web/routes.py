# from __main__ import app
from main.core.services import WeightCompleter
import main.web.web_controller as WebController
from flask import Flask, jsonify, request, json, Blueprint
from sqlalchemy import table
from main.core.models import *
from datetime import datetime

server_name = "sw502"
completer = WeightCompleter()
completer.initialize()

routes = Blueprint('routes', __name__)

# Root
@routes.route('/')
def home():
    return 'The weight completion service is running! :o)'


# Makes the service visible to UI
@routes.route('/info')
def info():
    return jsonify({
        'id': 'sw502_weight_completion',
        'name': 'Weight Completion',
        'version': '2020v1',
        'category': 1,
    })


# Get input fields for UI
@routes.route('/fields')
def fields():
    return WebController.make_fields(completer.avail_time_intervals)


# Get Readme
@routes.route('/readme')
def readme():
    return WebController.get_readme()


# Input received
@routes.route('/render', methods=["POST"])
def render():
    # If input has been entered, run the service
    date_from_form = request.form.get('date')
    interval_from_form = ''
    if date_from_form:
        interval_from_form = request.form.get(WebController.make_interval_name_from_date_string(date_from_form))
    isCompleted = request.form.get('completed')
    return WebController.render(date_from_form, interval_from_form, isCompleted, completer)

# This methods parses local OSM file and saves it into an database
# Commented to now allow access
# @app.route('/createnodes')
# def createnodes():
#    return WebController.parse_osm_and_save_to_db()
