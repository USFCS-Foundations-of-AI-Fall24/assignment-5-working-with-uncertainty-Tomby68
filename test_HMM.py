import unittest
import HMM


class MyTestCase(unittest.TestCase):
    def test_load(self):
        h = HMM.HMM()
        h.load("cat")
        print(h.emissions)
        print(h.transitions)


if __name__ == '__main__':
    unittest.main()
