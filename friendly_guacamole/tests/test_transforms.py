from unittest import TestCase
from biom import Table
from friendly_guacamole.transforms import AsDense
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal


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
