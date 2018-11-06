from pathlib import Path

from jubakit.loader.csv import CSVLoader
from jubakit.classifier import Schema, Dataset


def load_mails():
    """Create data set from CSV file.
    :return: dataset
    :rtype jubakit.classifier.Dataset
    """
    # Get a CSV path.
    current_dir = Path(__file__).parent
    csv_file_path = current_dir / 'mails.csv'

    # Load a CSV file.
    loader = CSVLoader(csv_file_path)

    # Define a Schema that defines types for each columns of the CSV file.
    # In this case, we are trying to classify a category of email from its subject.
    # It looks like we have to include all attribute of CSV to schema even if we only use specific attribute.
    schema = Schema(
        {
            'category': Schema.LABEL,
            'subject': Schema.STRING,
            'id': Schema.IGNORE,
            'from_address': Schema.IGNORE,
            'text': Schema.STRING,
            'html': Schema.IGNORE,
            'received_at': Schema.IGNORE,
            'attachments': Schema.IGNORE
        }
    )

    # Shuffle training data for better learning.
    dataset = Dataset(loader, schema).shuffle()

    return dataset
