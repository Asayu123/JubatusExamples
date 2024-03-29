from jubakit.classifier import Classifier

from client.settings import MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT
from client.main.mail_classifier.data.loader import load_mails

# Load a CSV file.
dataset = load_mails()

# Shuffle the dataset, as the dataset is sorted by label.
dataset = dataset.shuffle()

# Connect to an existing Jubatus service.
classifier = Classifier(MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT)

# Train the classifier with every data in the dataset.
for (idx, label) in classifier.train(dataset):
    # You can peek the datum being trained.
    print("Train: {0}".format(dataset[idx]))

# Classify using the same dataset.
use_softmax = True
for (idx, label, result) in classifier.classify(dataset, use_softmax):
    print("Classify: {0} (label: {1}, estimated: {2})".format(label == result[0][0], label, result[0][0]))
    for (est_label, est_score) in result:
        print("    Estimated Label: {0} ({1})".format(est_label, est_score))

classifier.stop()
