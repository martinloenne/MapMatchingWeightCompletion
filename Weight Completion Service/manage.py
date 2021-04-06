from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
import os

from app import app, db
from main.controller.routes import *
MIGRATION_DIR = os.path.join('.', 'main/data/migrations')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

migrate = Migrate(app, db, directory=MIGRATION_DIR)
manager = Manager(app)

manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    manager.run()