from sklearn.metrics.classification import classification_report

from jubakit.classifier import Classifier

from client.modules.helpers import k_fold_validation
from client.settings import MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT
from client.main.mail_classifier.data.loader import load_mails

# Load a CSV file.
dataset = load_mails()

# Connect to an existing Jubatus service.
classifier = Classifier(MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT)

# Execute training with K-fold validation.
true_labels, predicted_labels = k_fold_validation(
    service=classifier, dataset=dataset,
    n_splits=10,
)

# Show a classification report.
print(' === Summary of K-fold validation === ')
print(classification_report(true_labels, predicted_labels))

# Stop and clear the classifier.
classifier.stop()
classifier.clear()
