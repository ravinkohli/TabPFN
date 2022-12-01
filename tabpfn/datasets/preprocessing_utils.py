import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def remove_high_cardinality(X, y, categorical_mask, threshold=20):
    high_cardinality_mask = X.nunique() > threshold
    print("high cardinality columns: {}".format(X.columns[high_cardinality_mask * categorical_mask]))
    n_high_cardinality = sum(categorical_mask * high_cardinality_mask)
    X = X.drop(X.columns[categorical_mask * high_cardinality_mask], axis=1)
    print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    categorical_mask = [categorical_mask[i] for i in range(len(categorical_mask)) if not (high_cardinality_mask[i] and categorical_mask[i])]
    return X, y, categorical_mask, n_high_cardinality


def remove_pseudo_categorical(X, y):
    """Remove columns where most values are the same"""
    pseudo_categorical_cols_mask = X.nunique() < 10
    print("Removed {} columns with pseudo-categorical values on {} columns".format(sum(pseudo_categorical_cols_mask),
                                                                                   X.shape[1]))
    X = X.drop(X.columns[pseudo_categorical_cols_mask], axis=1)
    return X, y, sum(pseudo_categorical_cols_mask)


def remove_rows_with_missing_values(X, y):
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return X, y


def remove_missing_values(X, y, threshold=0.7, return_missing_col_mask=True):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    if return_missing_col_mask:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask), missing_cols_mask.values
    else:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask)


def balance(x, y):
    rng = np.random.RandomState(0)
    print("Balancing")
    print(x.shape)
    indices = [(y == i) for i in np.unique(y)]
    sorted_classes = np.argsort(
        list(map(sum, indices)))  # in case there are more than 2 classes, we take the two most numerous

    n_samples_min_class = sum(indices[sorted_classes[-2]])
    print("n_samples_min_class", n_samples_min_class)
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = (y == sorted_classes[-1])
    indices_second_class = (y == sorted_classes[-2])
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return x.iloc[total_indices], y

def preprocessing(X, y, categorical_indicator, categorical, regression, transformation=None):
    num_categorical_columns = sum(categorical_indicator)
    original_n_samples, original_n_features = X.shape
    le = LabelEncoder()
    if not regression:
        y = le.fit_transform(y)
    binary_variables_mask = X.nunique() == 2
    for i in range(X.shape[1]):
        if binary_variables_mask[i]:
            categorical_indicator[i] = True
    for i in range(X.shape[1]):
        if type(X.iloc[1, i]) == str:
            categorical_indicator[i] = True


    pseudo_categorical_mask = X.nunique() < 10
    n_pseudo_categorical = 0
    cols_to_delete = []
    for i in range(X.shape[1]):
        if pseudo_categorical_mask[i]:
            if not categorical_indicator[i]:
                n_pseudo_categorical += 1
                cols_to_delete.append(i)
    if not categorical:
        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                cols_to_delete.append(i)
    print("low card to delete")
    print(X.columns[cols_to_delete])
    print("{} low cardinality numerical removed".format(n_pseudo_categorical))
    X = X.drop(X.columns[cols_to_delete], axis=1)
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                             not i in cols_to_delete] # update categorical indicator
    print("Remaining categorical")
    print(categorical_indicator)
    X, y, categorical_indicator, num_high_cardinality = remove_high_cardinality(X, y, categorical_indicator, 20)
    print([X.columns[i] for i in range(X.shape[1]) if categorical_indicator[i]])
    X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y, 0.2)
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                             not missing_cols_mask[i]]
    if not regression:
        X, y = balance(X, y)
        y = le.fit_transform(y)
        assert len(X) == len(y)
        assert len(np.unique(y)) == 2
        assert np.max(y) == 1
    for i in range(X.shape[1]):
        if categorical_indicator[i]:
            X.iloc[:, i] = LabelEncoder().fit_transform(X.iloc[:, i])

    # if transformation is not None:
    #     assert regression
    #     y = transform_target(y, transformation)
    # else:
    #     print("NO TRANSFORMATION")

    return X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, num_categorical_columns, \
              n_pseudo_categorical, original_n_samples, original_n_features


