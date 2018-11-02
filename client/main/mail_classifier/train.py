from pathlib import Path
from jubakit.classifier import Classifier, Schema, Dataset
from jubakit.loader.csv import CSVLoader

# Get a script path.
current_dir = Path(__file__).parent
csv_file_path = current_dir / 'data/mails.csv'

# Load a CSV file.
loader = CSVLoader(csv_file_path)

# Define a Schema that defines types for each columns of the CSV file.
# In this case, we are trying to classify a category of email from its subject.
# It looks like we have to include all attribute of CSV to schema even if we only use specific attribute.
schema = Schema(
    {
        'category': Schema.LABEL,
        'subject': Schema.STRING,
        'id': Schema.NUMBER,
        'from_address': Schema.STRING,
        'text': Schema.STRING,
        'html': Schema.STRING,
        'received_at': Schema.STRING,
        'attachments': Schema.STRING
    }
)

# Shuffle training data for better learning.
dataset = Dataset(loader, schema).shuffle()

# Connect to an existing Jubatus service.
classifier = Classifier('127.0.0.1', 9199)  # TODO: Remove hard coding

# Train the classifier with every data in the dataset.
# TODO: Implement cross validation.
for (idx, label) in classifier.train(dataset):
    # You can peek the datum being trained.
    print("Train: {0}".format(dataset[idx]))

# Classify using the same dataset.
# TODO: Implement cross validation.
use_softmax = True
for (idx, label, result) in classifier.classify(dataset, use_softmax):
    print("Classify: {0} (label: {1}, estimated: {2})".format(label == result[0][0], label, result[0][0]))
    for (est_label, est_score) in result:
        print("    Estimated Label: {0} ({1})".format(est_label, est_score))

# Clear and Stop the classifier.
classifier.clear()
classifier.stop()

