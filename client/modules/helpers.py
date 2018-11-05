from sklearn.model_selection import StratifiedKFold


def k_fold_validation(service, n_splits, dataset):
    """ Execute K-fold validation with given data set and indices.
    WARNING: This method CLEAR trained models on Jubatus Server.
    :param service: An instance of Jubakit service.
    :param n_splits: Number of folds. Must be at least 2.
    :type n_splits: int
    :param dataset: An instance of Jubatus Dataset.
    :return: true_labels, predicted_labels
    :rtype:
    """

    true_labels = []
    predicted_labels = []

    # Setting up a K-fold validation parameters.
    labels = list(dataset.get_labels())
    skf = StratifiedKFold(n_splits=n_splits)
    train_test_indices = skf.split(labels, labels)

    for train_idx, test_idx in train_test_indices:
        # Clear the classifier (call `clear` RPC).
        service.clear()

        # Split the dataset to train/test dataset.
        (train_ds, test_ds) = (dataset[train_idx], dataset[test_idx])

        # Train the classifier using train dataset.
        for (idx, label) in service.train(train_ds):
            # You can peek records being trained.
            #print('train[{0}]: (label: {1}) => {2}'.format(idx, label, train_ds[idx]))
            pass

        # Test the classifier using test dataset.
        for (idx, label, result) in service.classify(test_ds):
            predicted_label = result[0][0]

            # You can peek records being classified.
            #print('classify[{0}]: (true: {1}, predicted: {2}) => {3}'.format(idx, label, pred_label, test_ds[idx]))

            # Store the result.
            true_labels.append(label)
            predicted_labels.append(predicted_label)

    service.clear()
    return true_labels, predicted_labels
