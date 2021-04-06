# This application has been developed by group SW502 as a part of the 5th semester
# in Software Engineering on Aalborg University (AAU).

# Please note: Files placed in the package "graphcompletionlib" are a clone of the GraphCompletion algorithm
# found here: https://github.com/hujilin1229/GraphCompletion
# However, a few modifications has been made to the algorithm in order to let adapt it to this project.
# These modifications are marked by "SW502" as far as possible.

from datetime import datetime
from flask import Flask
from flask_cors import CORS
import sys
from main.common.logger import log
from main.configuration.config import *
from main.core.models import *
from graphcompletionlib import gcrn_main_gcnn
from main.core.services import WeightCompleter

log("Starting app...")

if len(sys.argv) > 1 and sys.argv[1] == "prod":
    print("Loading production config")
    config = ProductionConfig()
else:
    print("Loading development config")
    config = DevelopmentConfig()

app = Flask(__name__)
app.config.from_object(config)
log(f'ENV is set to: {app.config["ENV"]}')

from main.web.routes import routes
app.register_blueprint(routes)

CORS(app)
db.init_app(app)


ctx = app.app_context()
ctx.push()


print("App is starting!")

ctx.pop()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
