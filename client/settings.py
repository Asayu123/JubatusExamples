# coding: UTF-8
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables to class variable.

MAIL_CLASSIFIER_HOST = os.getenv('MAIL_CLASSIFIER_HOST')
MAIL_CLASSIFIER_PORT = int(os.getenv('MAIL_CLASSIFIER_PORT'))

MAIL_RECOMMENDER_HOST = os.getenv('MAIL_RECOMMENDER_HOST')
MAIL_RECOMMENDER_PORT = int(os.getenv('MAIL_RECOMMENDER_PORT'))
