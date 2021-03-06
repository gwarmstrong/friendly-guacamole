from unittest import TestCase
from biom import Table
from friendly_guacamole.transforms import (
    AsDense, CLR, Rarefaction, RarefactionBIOM
)
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal


class AsDenseTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.test_table = Table(
            np.array([[1., 2, 3],
                      [0, 1, 0],
                      ]),
            observation_ids=['tax one', 'tax two'],
            sample_ids=['s1', 's2', 's3'],
        )

    def test_as_dense(self):
        t = AsDense()
        out = t.fit_transform(self.test_table)
        expected = pd.DataFrame([
            [1., 0.],
            [2, 1],
            [3, 0],
            ],
            columns=['tax one', 'tax two'],
            index=['s1', 's2', 's3'],
        )
        assert_frame_equal(expected, out)


class CLRTests(TestCase):
    def setUp(self):
        self.table = np.array([
            [1., 2., 3.],
            [0., 7., 0.],
        ])

    def test_clr_transform(self):
        t = CLR()
        X = t.fit_transform(self.table)
        g = 2 * (3 ** (1 / 3))
        expected = np.array([
            [np.log(2 / g), np.log(3 / g), np.log(4 / g), ],
            [np.log(1 / 2), np.log(8 / 2), np.log(1 / 2), ],
        ])
        assert_array_almost_equal(expected, X)

    def test_clr_transform_pandas(self):
        t = CLR()
        X = t.fit_transform(pd.DataFrame(self.table))
        g = 2 * (3 ** (1 / 3))
        expected = np.array([
            [np.log(2 / g), np.log(3 / g), np.log(4 / g), ],
            [np.log(1 / 2), np.log(8 / 2), np.log(1 / 2), ],
        ])
        assert_array_almost_equal(expected, X)


class RarefactionTests(TestCase):
    def setUp(self):
        self.test_table = np.array([
            [1., 2., 3., ],
            [0., 8., 9., ],
        ])

    def test_rarefaction(self):
        t = Rarefaction(depth=3)
        X = t.fit_transform(self.test_table)
        expected_row_sums = [3, 3]
        observed_row_sums = X.sum(axis=1)
        assert_array_almost_equal(expected_row_sums, observed_row_sums)

    def test_rarefaction_reduces_features(self):
        t = Rarefaction(depth=1)
        X = t.fit_transform(self.test_table)
        observed_shape = X.shape
        self.assertEqual(2, observed_shape[0])
        self.assertLessEqual(observed_shape[1], 2)


class RarefactionBIOMTests(TestCase):
    def setUp(self):
        self.test_table = Table(
            np.array([[5., 2, 0],
                      [0, 7, 9],
                      ]),
            observation_ids=['tax one', 'tax two'],
            sample_ids=['s1', 's2', 's3'],
        )

    def test_rarefaction(self):
        t = RarefactionBIOM(3)
        t = t.fit(self.test_table)
        X = t.transform(self.test_table)
        expected_row_sums = [3, 3, 3]
        observed_row_sums = X.sum(axis='sample')
        assert_array_almost_equal(expected_row_sums, observed_row_sums)
