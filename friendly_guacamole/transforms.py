import tempfile
from contextlib import contextmanager
import numpy as np
from sklearn.base import TransformerMixin
from scipy.spatial.distance import cdist
from biom.util import biom_open
from skbio.stats.composition import clr
from skbio.stats import subsample_counts
from skbio.stats.ordination import pcoa
from friendly_guacamole.utils import as_dense
import pandas as pd
from unifrac import ssu, faith_pd


class AsDense(TransformerMixin):
    """
    converts a biom.Table into a pd.DataFrame
    """

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : biom.Table
            feature table
        y : None
            ignored

        Returns
        -------
        self : object
            Fitted transformer
        """
        return self

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : biom.Table
            feature table
        y : None
            ignored

        Returns
        -------
        X_new : pd.DataFrame
            Transformed data

        """
        return as_dense(X)


class _unifracMixin:

    def fit(self, X, y=None):
        """

        X : biom.Table
            feature table
        y : None
            ignored

        Returns
        -------
        self : object
            fitted transformer

        """
        self.table = X
        return self

    @contextmanager
    def hdf5_table(self, X):
        with tempfile.NamedTemporaryFile() as f:
            with biom_open(f.name, 'w') as b:
                X.to_hdf5(b, "merged")
                yield f


class UniFrac(TransformerMixin, _unifracMixin):
    """
    computes the UniFrac distance on a biom.Table

    Parameters
    ----------
    tree_path : string
        Path to a phylogeny containing all IDs in the candidate tables
    unifrac_method : string
        UniFrac method to use. See `unifrac` package.

    """

    def __init__(self, tree_path, unifrac_method='unweighted'):
        self.tree_path = tree_path
        self.unifrac_method = unifrac_method
        self.table = None

    def transform(self, X):
        """

        X : biom.Table
            feature table
        y : None
            ignored

        Returns
        -------
        X_new : pd.DataFrame
            Transformed data

        """
        sub_dm = self._get_distances(X)
        return sub_dm

    def _get_distances(self, X):
        dm = self._get_distance_matrix(X)
        sub_dm = self._extract_sub_matrix(X, dm)
        return sub_dm

    def _extract_sub_matrix(self, X, dm):
        # get indices of test ID's
        X_idx = [dm.index(name) for name in X.ids('sample')]
        # get indices of table ID's
        ref_idx = [dm.index(name) for name in self.table.ids('sample')]
        # extract sub-distance matrix
        idxs = np.ix_(X_idx, ref_idx)
        sub_dm = dm.data[idxs]
        return sub_dm

    def _get_distance_matrix(self, X):
        """
        computes UniFrac distances with the fitted samples

        Parameters
        ----------
        X : biom.Table
            new samples

        Returns
        -------
        dm : DistanceMatrix
            distances from old samples to new samples

        """
        # TODO one problem with this approach is that
        #  if any samples in X overlap self.table, the counts will
        #  be doubled
        merged_table = self.table.merge(X)
        with self.hdf5_table(merged_table) as f:
            dm = ssu(f.name, self.tree_path,
                     unifrac_method=self.unifrac_method,
                     variance_adjust=False,
                     alpha=1.0,
                     bypass_tips=False,
                     threads=1,
                     )
        return dm


class FaithPD(TransformerMixin, _unifracMixin):

    def __init__(self, tree_path):
        self.tree_path = tree_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        with self.hdf5_table(X) as f:
            fpd = faith_pd(f.name, self.tree_path)
        return fpd.to_frame()


class RarefactionBIOM(TransformerMixin):
    """
    rarefies a biom.Table

    Parameters
    ----------
    depth : int
        rarefaction depth
    replace : bool, optional
        indicates whether sampling should
         with replacement. default=False.

    """

    def __init__(self, depth, replace=False):
        self.depth = depth
        self.replace = replace
        self.index = None
        self.features = None

    def fit(self, X, y=None):
        """

        X : biom.Table
            feature table
        y : None
            ignored

        Returns
        -------
        self : object
            fitted transformer

        """
        self.index = X.subsample(n=self.depth,
                                 with_replacement=self.replace)
        self.features = self.index.ids('observation')
        return self

    def transform(self, X):
        """ rarefies a biom.Table

        Parameters
        ----------
        X : biom.Table
            feature table

        Returns
        -------
        X_new : biom.Table
            rarefied table

        """
        # TODO There is an unaccounted for  edge case here,
        #  when samples have fewer counts than the depth
        X = X.filter(ids_to_keep=self.features, axis='observation',
                     inplace=False)
        index_ids = set(self.index.ids('sample'))
        known_ids = [id_ for id_ in X.ids('sample') if id_ in index_ids]
        unknown_ids = [id_ for id_ in X.ids('sample') if id_ not in index_ids]
        known_counts = self.index.filter(ids_to_keep=known_ids, axis='sample',
                                         inplace=False
                                         )
        unknown_counts = X.filter(ids_to_keep=unknown_ids, axis='sample',
                                  inplace=False
                                  )
        unknown_counts = unknown_counts.subsample(
            n=self.depth,
            with_replacement=self.replace
        )
        # TODO arghhh this really needs unit tests
        self.index.merge(unknown_counts)
        all_counts = known_counts.merge(unknown_counts)
        all_counts.sort_order(X.ids('sample'), axis='sample')
        return all_counts


class Rarefaction(TransformerMixin):
    """
    Rarefies an array-like

    Parameters
    ----------
    depth : int
        rarefaction depth
    replace : bool, optional
        indicates whether sampling should
         with replacement. default=False.

    """

    def __init__(self, depth, replace=False):
        self.depth = depth
        self.replace = replace
        self.idx = None

    def fit(self, X, y=None):
        """

        X : array-like
            feature table
        y : None
            ignored

        Returns
        -------
        self : object
            fitted transformer

        """
        X, self.idx = self._find_nonzero_idx(X)
        return self

    def transform(self, X, y=None):
        """ rarefies a feature table
        Caution: this will return different results for the same sample

        Parameters
        ----------
        X : array-like
            feature table

        Returns
        -------
        X_new : array-like
            rarefied table

        """
        if isinstance(X, pd.DataFrame):
            idx = np.array([True] * len(X.columns))
            idx[self.idx[:, 1]] = False
            X = X.loc[:, idx]
        else:
            X = np.delete(X, self.idx, axis=1)
        X = self._subsample(X)

        return X

    def _find_nonzero_idx(self, X):
        X = self._subsample(X)
        # remove columns with zero counts
        row_sums = X.sum(axis=0, keepdims=True)
        idx = np.argwhere(row_sums == 0)
        return X, idx

    def _subsample(self, X):
        X = X.astype(int)
        X_out = list()
        iter_var = X.values if isinstance(X, pd.DataFrame) else X
        for row in iter_var:
            new_X = subsample_counts(row, n=self.depth, replace=self.replace)
            X_out.append(new_X)
        X = np.vstack(X_out)
        return X


class CLR(TransformerMixin):
    """
    performs the center log ratio transform with pseudo-count

    Parameters
    ----------
    pseudocount : int, optional
        Count to add to every entry of the table

    """

    def __init__(self, pseudocount=1):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        """

        X : array-like
            feature table
        y : None
            ignored

        Returns
        -------
        self : object
            fitted transformer

        """
        return self

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : pd.DataFrame
            feature table
        y : None
            ignored

        Returns
        -------
        X_new : pd.DataFrame
            Transformed data

        """
        transfored_data = clr(X + self.pseudocount)
        if X.shape[0] == 1:
            transfored_data = transfored_data.reshape(1, -1)

        return transfored_data


class PCoA(TransformerMixin):
    def __init__(self, metric='precomputed'):
        """Performas a PCoA on the data

        Parameters
        ----------
        metric : str, default='precomputed'
            metric to compute PCoA on. If 'precomputed', a distance matrix is
            expected

        """
        self.metric = metric
        self.embedding_ = None
        self.ordination_results_ = None

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : array-like
            Feature table or distance matrix
        y : None
            ignored

        Returns
        -------
        self
            fitted pcoa

        """
        # TODO validation on X
        if self.metric != 'precomputed':
            X = cdist(X, X, metric=self.metric)

        self.ordination_results_ = pcoa(X)
        self.embedding_ = self.ordination_results_.samples

        return self

    def fit_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : array-like
            Feature table or distance matrix

        Returns
        -------
        array-like
            embeddings of the samples
        """
        self.fit(X, y)
        return self.embedding_


class FilterSamples(TransformerMixin):

    def __init__(self, min_count=0):
        """
        Filters samples

        Parameters
        ----------
        min_count : int
            Minimum number of feature counts neede dto retain sample

        """
        self.min_count = min_count

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : biom.Table

        Returns
        -------
        self
            fitted transformer

        """
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : biom.Table

        Returns
        -------
        biom.Table
            The filtered table.

        """
        sample_counts = X.sum(axis='sample')
        insufficient_counts = (sample_counts < self.min_count)
        ids_to_remove = set(X.ids('sample')[insufficient_counts])
        return X.filter(ids_to_remove, invert=True, inplace=False)
