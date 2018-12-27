from unittest import TestCase

# Dependencies.
from jubakit.classifier import Classifier as JubatusClassifier

from client.settings import MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT
from client.main.mail_classifier.data.loader import load_mails
from client.modules.helpers import k_fold_validation

# Dependencies for test.
from sklearn.metrics.classification import classification_report


class TestMailClassifier(TestCase):
    _estimator_class = JubatusClassifier  # An RPC client to the estimator which located on a remote server.

    def setUp(self):
        self.estimator = self._estimator_class(MAIL_CLASSIFIER_HOST, MAIL_CLASSIFIER_PORT)
        self.dataset = load_mails()  # A stream will be closed if all data has been yielded, so load it every test.
        self.dataset.shuffle()

    def tearDown(self):
        # In this case, we are going to clear the learned model on the remote server every test.
        self.estimator.clear()
        self.estimator.stop()

    def test_model_performance(self):
        # Execute a K-fold Cross-validation.
        true_labels, predicted_labels = k_fold_validation(
            service=self.estimator, dataset=self.dataset,
            n_splits=10,
        )

        # Print result to stdout for human readability and logging.
        result_str = classification_report(true_labels, predicted_labels)
        print(result_str)

        # Check automatically whether the model qualifies the target performance.
        result = classification_report(true_labels, predicted_labels, output_dict=True)

        test_patterns = [
            ('job f1-score', result['job']['f1-score'], 0.95),
            ('person f1-score', result['person']['f1-score'], 0.95),
        ]

        # Evaluate the scores.
        for metric_name, actual_score, target_score in test_patterns:
            with self.subTest(metric=metric_name):
                self.assertTrue(
                    actual_score >= target_score,
                    msg="{0} doensn't qualify the target score: actual={1}, target={2} ".format(
                        metric_name, actual_score, target_score
                    )
                )
