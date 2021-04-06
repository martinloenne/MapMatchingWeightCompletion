from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))

# Base configuration
class Config:
    CORS_HEADERS = 'Content-Type'
    FLASK_ENV = None
    ENV = None
    DEBUG = None
    TESTING = None
    SQLALCHEMY_DATABASE_URI = None
    TEST = None

class ProductionConfig(Config):
    def __init__(self):
        load_dotenv(path.join(basedir, 'prod.env'))
        Config.FLASK_ENV = 'production'
        Config.ENV = 'production'
        Config.DEBUG = False
        Config.TESTING = False
        Config.SQLALCHEMY_DATABASE_URI = environ.get('502_CONNECTION_STRING')
        Config.SQLALCHEMY_BINDS = {'rou' : environ.get('ROU_CONNECTION_STRING')}


class DevelopmentConfig(Config):
    def __init__(self):
        load_dotenv(path.join(basedir, 'dev.env'))
        Config.FLASK_ENV = 'development'
        Config.ENV = 'development'
        Config.DEBUG = True
        Config.TESTING = True
        Config.SQLALCHEMY_DATABASE_URI = environ.get('LOCALHOST_CONNECTION_STRING')
        Config.SQLALCHEMY_BINDS = {'rou': environ.get('ROU_CONNECTION_STRING')}

