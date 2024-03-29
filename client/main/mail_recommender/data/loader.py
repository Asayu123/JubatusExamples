from pathlib import Path

from jubakit.loader.csv import CSVLoader
from jubakit.recommender import Schema, Dataset


def load_mails(path):
    """Create data set from CSV file.
    :param path: An absolute path of csv file.
    :type path: str
    :return: dataset
    :rtype jubakit.classifier.Dataset
    """

    # Load a CSV file.
    loader = CSVLoader(path)

    # Define a Schema that defines types for each columns of the CSV file.
    # In this case, we are trying to classify a category of email from its subject.
    # It looks like we have to include all attribute of CSV to schema even if we only use specific attribute.
    schema = Schema(
        {
            'category': Schema.IGNORE,
            'subject': Schema.STRING,
            'id': Schema.ID,
            'from_address': Schema.IGNORE,
            'text': Schema.STRING,
            'received_at': Schema.IGNORE,
            'attachments': Schema.STRING
        }
    )
    # TODO: Append a validation logic for attachments attribute. (It must be space separated strings.)

    # Shuffle training data for better learning.
    dataset = Dataset(loader, schema)

    return dataset
