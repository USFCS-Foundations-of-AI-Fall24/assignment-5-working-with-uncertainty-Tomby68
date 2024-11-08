import unittest
import HMM


class MyTestCase(unittest.TestCase):
    def test_load(self):
        h = HMM.HMM()
        h.load("partofspeech")
        print(h.emissions)
        print(h.transitions)

    def test_generate(self):
        h = HMM.HMM()
        h.load("partofspeech")
        seq = h.generate(20)
        print(seq)


if __name__ == '__main__':
    unittest.main()
