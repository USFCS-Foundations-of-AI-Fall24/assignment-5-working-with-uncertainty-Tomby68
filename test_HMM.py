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

    def test_forward(self):
        h = HMM.HMM()
        h.load("cat")
        seq = h.generate(10)
        last_state = h.forward(seq)
        print(f"Prediction: {last_state}\nActual: {seq.stateseq[-1]}\n")


if __name__ == '__main__':
    unittest.main()
