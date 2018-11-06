from jubakit.recommender import Recommender

from client.settings import MAIL_RECOMMENDER_HOST, MAIL_RECOMMENDER_PORT
from client.main.mail_recommender.data.loader import load_mails

# Load a CSV file.
dataset = load_mails()

# Shuffle the dataset, as the dataset is sorted by label.
dataset = dataset.shuffle()

# Connect to an existing Jubatus service.
service = Recommender(MAIL_RECOMMENDER_HOST, MAIL_RECOMMENDER_PORT)

# Train the recommender with every data in the dataset.
for (idx, row_id, success) in service.update_row(dataset):
    pass

# Calculate the similarity in recommender model from datum and display top-2 similar items.
print('{0}\n recommend similar documents from datum \n{1}'.format('-'*60, '-'*60))
for (idx, row_id, result) in service.similar_row_from_datum(dataset, size=3):
  if idx % 10 == 0:
    print('DocumentID {0} is similar to : {1} (score:{2:.3f}), {3} (score:{4:.3f})'.format(
          result[0].id, result[1].id, result[1].score, result[2].id, result[2].score))

# Clear and Stop the classifier. TODO: Remove this if you want to persist a trained model.
service.clear()
service.stop()
