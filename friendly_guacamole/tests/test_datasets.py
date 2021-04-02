from unittest import TestCase
from tempfile import TemporaryDirectory
from friendly_guacamole.datasets import (
    KeyboardDataset,
    DietInterventionStudy,
)


class TestRealDataSets(TestCase):
    def test_keyboard(self):
        with TemporaryDirectory() as td:
            kbd = KeyboardDataset(td)
            kbd['table']
            kbd['metadata']
            kbd['tree']

    def test_diet(self):
        with TemporaryDirectory() as td:
            dis = DietInterventionStudy(td)
            dis['table']
            dis['metadata']
            dis['tree']
