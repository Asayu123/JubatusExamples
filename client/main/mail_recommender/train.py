from jubakit.recommender import Recommender

from client.settings import MAIL_RECOMMENDER_HOST, MAIL_RECOMMENDER_PORT
from client.main.mail_recommender.data.loader import load_mails

from pathlib import Path

# Load a CSV file.
dataset = load_mails(path=Path(__file__).parent/'data'/'job_mails.csv')

# Shuffle the dataset, as the dataset is sorted by label.
dataset = dataset.shuffle()

# Connect to an existing Jubatus service.
service = Recommender(MAIL_RECOMMENDER_HOST, MAIL_RECOMMENDER_PORT)

# Train the recommender with every data in the dataset.
for (idx, row_id, success) in service.update_row(dataset):
    pass

# Clear and Stop the recommender. TODO: Remove this if you want to persist a trained model.
service.save(name='job_offers')
