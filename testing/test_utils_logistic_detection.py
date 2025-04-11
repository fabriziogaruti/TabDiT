import json
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path

"""SDMetrics utils to be used across all the project."""
import warnings
from collections import Counter
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""Base class for Machine Learning Detection metrics for single table datasets."""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
# from sdmetrics.utils import HyperTransformer: Use the fixed version above with OneHotEncoder variable handle_unknown="ignore"

"""scikit-learn based DetectionMetrics for single table datasets."""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
# from sdmetrics.single_table.detection.base import DetectionMetric: Use the fixed version above with HyperTransformer using OneHotEncoder variable handle_unknown="ignore"

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel, delayed

from loguru import logger





def nested_attrs_meta(nested):
    """Metaclass factory that defines a Metaclass with a dynamic attribute name."""

    class Metaclass(type):
        """Metaclass which pulls the attributes from a nested object using properties."""

        def __getattr__(cls, attr):
            """If cls does not have the attribute, try to get it from the nested object."""
            nested_obj = getattr(cls, nested)
            if hasattr(nested_obj, attr):
                return getattr(nested_obj, attr)

            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{attr}'")

        @property
        def name(cls):
            return getattr(cls, nested).name

        @property
        def goal(cls):
            return getattr(cls, nested).goal

        @property
        def max_value(cls):
            return getattr(cls, nested).max_value

        @property
        def min_value(cls):
            return getattr(cls, nested).min_value

    return Metaclass


def get_frequencies(real, synthetic):
    """Get percentual frequencies for each possible real categorical value.

    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.

    Args:
        real (list):
            A list of hashable objects.
        synthetic (list):
            A list of hashable objects.

    Yields:
        tuble[list, list]:
            The observed and expected frequencies (as a percent).
    """
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            warnings.warn(f'Unexpected value {value} in synthetic data.')
            real[value] += 1e-6  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))  # noqa: PD011
        f_exp.append(real[value] / sum(real.values()))  # noqa: PD011

    return f_obs, f_exp


def get_cardinality_distribution(parent_column, child_column):
    """Compute the cardinality distribution of the (parent, child) pairing.

    Args:
        parent_column (pandas.Series):
            The parent column.
        child_column (pandas.Series):
            The child column.

    Returns:
        pandas.Series:
            The cardinality distribution.
    """
    child_df = pd.DataFrame({'child_counts': child_column.value_counts()})
    cardinality_df = pd.DataFrame({'parent': parent_column}).join(
        child_df, on='parent').fillna(0)

    return cardinality_df['child_counts']


def is_datetime(data):
    """Determine if the input is a datetime type or not.

    Args:
        data (pandas.DataFrame, int or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    return (
        pd.api.types.is_datetime64_any_dtype(data)
        or isinstance(data, pd.Timestamp)
        or isinstance(data, datetime)
    )


class HyperTransformer():
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a set of transforms to transform one or
    more columns based on each column's data type.
    """

    column_transforms = {}
    column_kind = {}

    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                self.column_transforms[field] = {'mean': pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'value{i}' for i in range(np.shape(out)[1])])
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                data[field] = pd.Series(integers, index=data.index)
                data[field] = data[field].fillna(transform_info['mean'])

        return data

    def fit_transform(self, data):
        """Fit and transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        self.fit(data)
        return self.transform(data)


def get_columns_from_metadata(metadata):
    """Get the column info from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        dict:
            The columns metadata.
    """
    if 'fields' in metadata:
        return metadata['fields']

    if 'columns' in metadata:
        return metadata['columns']

    return []


def get_type_from_column_meta(column_metadata):
    """Get the type of a given column from the column metadata.

    Args:
        column_metadata (dict):
            The column metadata.

    Returns:
        string:
            The column type.
    """
    if 'type' in column_metadata:
        return column_metadata['type']

    if 'sdtype' in column_metadata:
        return column_metadata['sdtype']

    return ''


def get_alternate_keys(metadata):
    """Get the alternate keys from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        list:
            The list of alternate keys.
    """
    alternate_keys = []
    for alternate_key in metadata.get('alternate_keys', []):
        if type(alternate_key) is list:
            alternate_keys.extend(alternate_key)
        else:
            alternate_keys.append(alternate_key)

    return alternate_keys




class DetectionMetric(SingleTableMetric):
    """Base class for Machine Learning Detection based metrics on single tables.

    These metrics build a Machine Learning Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'SingleTable Detection'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _fit_predict(X_train, y_train, X_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        This builds a Machine Learning Classifier that learns to tell the synthetic
        data apart from the real data, which later on is evaluated using Cross Validation.

        The output of the metric is one minus the average ROC AUC score obtained.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            float:
                One minus the ROC AUC Cross Validation Score obtained by the classifier.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata)

        if metadata is not None and 'primary_key' in metadata:
            transformed_real_data = real_data.drop(metadata['primary_key'], axis=1)
            transformed_synthetic_data = synthetic_data.drop(metadata['primary_key'], axis=1)

        else:
            transformed_real_data = real_data
            transformed_synthetic_data = synthetic_data

        # logger.debug(f'data shape : real: {transformed_real_data.shape}, syn: {transformed_synthetic_data.shape}')
        # logger.debug(f'columns : {transformed_real_data.columns}')
        ht = HyperTransformer()
        transformed_real_data = ht.fit_transform(transformed_real_data)
        transformed_synthetic_data = ht.transform(transformed_synthetic_data)
        # logger.debug(f'transformed data shape : real: {transformed_real_data.shape}, syn: {transformed_synthetic_data.shape}')
        # logger.debug(f'columns : {transformed_real_data.columns}')
        transformed_real_data = transformed_real_data.to_numpy()
        transformed_synthetic_data = transformed_synthetic_data.to_numpy()
        X = np.concatenate([transformed_real_data, transformed_synthetic_data])
        y = np.hstack([
            np.ones(len(transformed_real_data)), np.zeros(len(transformed_synthetic_data))
        ])
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan

        # logger.debug(f'running logistic detection with model : {cls.__name__}')
        # logger.debug(f'input data shape : {X.shape}, labels shape : {y.shape}')
        try:
            scores = []
            kf = StratifiedKFold(n_splits=2, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                y_pred = cls._fit_predict(X[train_index], y[train_index], X[test_index])
                roc_auc = roc_auc_score(y[test_index], y_pred)

                scores.append(max(0.5, roc_auc) * 2 - 1)

            # logger.debug(f'output scores are : {scores}, final res is {1 - np.mean(scores)}')
            return 1 - np.mean(scores)
        except ValueError as err:
            raise IncomputableMetricError(f'DetectionMetric: Unable to be fit with error {err}')

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                Simply returns `raw_score`.
        """
        return super().normalize(raw_score)


class ScikitLearnClassifierDetectionMetric(DetectionMetric):
    """Base class for Detection metrics build using Scikit Learn Classifiers.

    The base class for these metrics makes a prediction using a scikit-learn
    pipeline which contains a SimpleImputer, a RobustScaler and finally
    the classifier, which is defined in the subclasses.
    """

    name = 'Scikit-Learn Detection'

    @staticmethod
    def _get_classifier():
        """Build and return an instance of a scikit-learn Classifier."""
        raise NotImplementedError()

    @classmethod
    def _fit_predict(cls, X_train, y_train, X_test):
        """Fit a pipeline to the training data and then use it to make prediction on test data."""
        model = Pipeline([
            ('imputer', SimpleImputer()),
            ('scalar', RobustScaler()),
            ('classifier', cls._get_classifier()),
        ])
        model.fit(X_train, y_train)

        return model.predict_proba(X_test)[:, 1]


class LogisticDetection(ScikitLearnClassifierDetectionMetric):
    """ScikitLearnClassifierDetectionMetric based on a LogisticRegression.

    This metric builds a LogisticRegression Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.
    """

    name = 'LogisticRegression Detection'

    @staticmethod
    def _get_classifier():
        # TODO: max_iter=1000, 5000
        return LogisticRegression(solver='lbfgs', max_iter=5000)


class RandomForestDetection(ScikitLearnClassifierDetectionMetric):
    """ScikitLearnClassifierDetectionMetric based on a RandomForest.

    This metric builds a RandomForest Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.
    """

    name = 'RandomForest Detection'

    @staticmethod
    def _get_classifier():
        return RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=25, oob_score=False)


def report_logistic(
    real_child, syn_child, 
    real_parent=None, syn_parent=None,
    join_on=None,
    seed=None, 
    verbose=True, 
    classifier="randomforest",
    ):
    # TODO: attenzione setta il seed e non lo ripristina!
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    if classifier == "logistic":
        lm = LogisticDetection()
    elif classifier == "randomforest":
        lm = RandomForestDetection()
    else:
        raise ValueError(f'Invalid classifier: "{classifier}"')

    if real_parent is not None and syn_parent is not None:
        real_flat = real_child.merge(real_parent, on=join_on)
        syn_flat = syn_child.merge(syn_parent, on=join_on)

    data = {}
    data['child'] = lm.compute(real_child.drop(join_on, axis=1), syn_child.drop(join_on, axis=1))
    if real_parent is not None and syn_parent is not None:
        data['parent'] = lm.compute(real_parent.drop(join_on, axis=1), syn_parent.drop(join_on, axis=1))
        data['merged'] = lm.compute(real_flat.drop(join_on, axis=1), syn_flat.drop(join_on, axis=1))

    if verbose:
        logger.info(f"LogisticDetection Model: {classifier}")
        logger.info(f"LD for children: {data['child']}")
        if real_parent is not None and syn_parent is not None:
            logger.info(f"LD for parents: {data['parent']}")
            logger.info(f"LD for merged: {data['merged']}")
    return data

def get_comp_samp(tables, join_on, seed, n=1000):
    parent_comp_samp = tables["parent"][join_on].sample(n=n, random_state=seed)
    parent_comp_samp = tables["parent"][tables["parent"][join_on].isin(parent_comp_samp.tolist())]
    child_comp_samp = tables["child"][tables["child"][join_on].isin(parent_comp_samp[join_on].tolist())]
    return parent_comp_samp, child_comp_samp





def wrap_metric_compute(real_child, syn_child, seed, ld_seed, join_on, LD_CLASSIFIER, real_parent=None, syn_parent=None):
    return dict(
            sample_seed=seed,
            ld_seed=ld_seed,
            stats=report_logistic(
                real_child=real_child,
                syn_child=syn_child,
                real_parent=real_parent,
                syn_parent=syn_parent,
                join_on=join_on,
                seed=ld_seed,
                verbose=True,
                classifier=LD_CLASSIFIER,
            )
        )


def print_statistics_ld(dataframe_tuple, join_on, output_dir, nr_seed_split_dataset=1, nr_seed_train_ld=1):
    ld_report_file = Path(output_dir) / 'ld_statistics.json'
    real_child, syn_child, real_parent, syn_parent = dataframe_tuple
    full_ld_data = {}

    for ld_classifier in ["logistic", "randomforest"]:
        ld_report_data = []

        # TODO: possibili seed di split train-generazione
        for seed in range(nr_seed_split_dataset):
            logger.info(f"{datetime.now()} ld_classifier: {ld_classifier} seed: {seed}")

            with Parallel(n_jobs=nr_seed_train_ld) as parallel:
                out = parallel(delayed(wrap_metric_compute)(
                    real_child, syn_child, seed, ld_seed, join_on, ld_classifier, real_parent, syn_parent
                    ) for ld_seed in range(nr_seed_train_ld))
                ld_report_data.extend(out)
            ld_report_file.write_text(json.dumps(ld_report_data, indent=4))

        full_data = {}

        syn_child_ld = []
        for lrd in ld_report_data:
            syn_child_ld.append(lrd["stats"]["child"])
        full_data["syn_child_ld_mean"] = np.mean(syn_child_ld)
        if len(syn_child_ld) > 1:
            full_data["syn_child_ld_std"] = np.std(syn_child_ld, ddof=1)

        if real_parent is not None and syn_parent is not None:
            syn_parent_ld = []
            syn_merged_ld = []
            for lrd in ld_report_data:
                syn_parent_ld.append(lrd["stats"]["parent"])
                syn_merged_ld.append(lrd["stats"]["merged"])
            full_data["syn_parent_ld_mean"] = np.mean(syn_parent_ld)
            full_data["syn_merged_ld_mean"] = np.mean(syn_merged_ld)
            if len(syn_child_ld) > 1:
                full_data["syn_parent_ld_std"] = np.std(syn_parent_ld, ddof=1)
                full_data["syn_merged_ld_std"] = np.std(syn_merged_ld, ddof=1)

        full_data["raw"] = ld_report_data
        logger.info(full_data)
        ld_report_file.write_text(json.dumps(full_data, indent=4))
        full_ld_data[ld_classifier] = full_data

    ld_report_file.write_text(json.dumps(full_ld_data, indent=4))





if __name__ == "__main__":
    path_dir = "/home/andrej/checkpoints/bbpm_categorizzato_18/decode_DIT_v13/big3_dit_digit_decoder_beta5_tabsyn_var1_1M_50_pad_s50_cp/M1"
    join_on = "user"

    output_dir = Path("./")
    BBPM_EXP = Path(path_dir)

    nr_seed_split_dataset = 1
    nr_seed_train_ld = 2
    limit_test_data = True

    # TODO: read also parent data if exists
    real_parent = None
    syn_parent = None
    real_child = pd.read_csv(BBPM_EXP / "true_trans.csv")
    syn_child = pd.read_csv(BBPM_EXP / "generated_trans.csv")
    logger.info(f"Test dataset shape: real: {real_child.shape}, generated: {syn_child.shape}")
    logger.info(f"Nr users real: {real_child[join_on].nunique()}, generated: {syn_child[join_on].nunique()}")

    if limit_test_data:
        logger.info("Sampling dataset")
        real_user_unique = pd.Series(real_child[join_on].unique()).sample(2000)
        syn_user_unique = pd.Series(syn_child[join_on].unique()).sample(2000)
        real_child = real_child[real_child[join_on].isin(real_user_unique.tolist())]
        syn_child = syn_child[syn_child[join_on].isin(syn_user_unique.tolist())]
        logger.info(f"Test dataset shape: real: {real_child.shape}, generated: {syn_child.shape}")
        logger.info(f"Nr users real: {real_child[join_on].nunique()}, generated: {syn_child[join_on].nunique()}")

    dataframe_tuple = (real_child, syn_child, real_parent, syn_parent)

    print_statistics_ld(dataframe_tuple, join_on, output_dir, nr_seed_train_ld=2)

