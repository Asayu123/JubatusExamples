from pathlib import Path

from jubakit.loader.csv import CSVLoader
from jubakit.classifier import Schema, Dataset


def load_mails():
    """Create data set from CSV file.
    :return: dataset
    :rtype jubakit.classifier.Dataset
    """
    # Get a CSV path.
    root = Path(__file__).parent.parent
    csv_file_path = root / 'data/mails.csv'

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

    return dataset
