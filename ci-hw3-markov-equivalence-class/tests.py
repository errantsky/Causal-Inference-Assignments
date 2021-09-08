import unittest
from mec_count.main import SizeMEC, get_essential_size
import numpy as np
from networkx import from_numpy_matrix, DiGraph


class TestEssentials(unittest.TestCase):
    def testEssential1(self):
        essential = from_numpy_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
            create_using=DiGraph(),
        )

        self.assertEqual(get_essential_size(essential), 2)


class TestSizeMEC(unittest.TestCase):
    def test_u56(self):
        uccg = from_numpy_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            )
        )

        self.assertEqual(SizeMEC(uccg), 13)

    def test_u56_prime(self):
        uccg = from_numpy_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0],
                ]
            )
        )

        self.assertEqual(SizeMEC(uccg), 12)

    def test_u57(self):
        uccg = from_numpy_matrix(
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            )
        )

        self.assertEqual(SizeMEC(uccg), 14)

    def test_u57_prime(self):
        uccg = from_numpy_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 1],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            )
        )

        self.assertEqual(SizeMEC(uccg), 30)


if __name__ == "__main__":
    unittest.main()
