from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.classification import classification_report

from client.modules.wrapper import ExtendedClassifier
from client.settings import MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT
from client.main.mail_classifier.data.loader import load_mails

# Load a CSV file.
dataset = load_mails()

# Setting up a K-fold validation parameters.
labels = list(dataset.get_labels())
skf = StratifiedKFold(n_splits=10)
train_test_indices = skf.split(labels, labels)

# Connect to an existing Jubatus service.
classifier = ExtendedClassifier(MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT)

# Execute training with K-fold validation.
true_labels, predicted_labels = classifier.k_fold_validation(
    dataset=dataset, train_test_indices=train_test_indices
)

# Stop and clear the classifier.
classifier.stop()
classifier.clear()

# Show a classification report.
print(classification_report(true_labels, predicted_labels))
