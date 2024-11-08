

import random
import argparse
import codecs
import os
import numpy
import sys

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

def parse_probability_contents(contents):
    result = {}
    for line in contents.split('\n'):
        line = line.split(' ')
        if len(line) == 3:
            if line[0] in result:
                result[line[0]][line[1]] = float(line[2])
            else:
                result[line[0]] = {line[1]: float(line[2])}
    return result

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        try:
            with open(basename + ".emit") as f: emissions = f.read()
            with open(basename + ".trans") as f: transitions = f.read()
        except FileNotFoundError:
            print("Error: basename does not correspond to .emit/.trans files")
        self.emissions = parse_probability_contents(emissions)
        self.transitions = parse_probability_contents(transitions)

   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        trans_seq = []
        em_seq = []
        if n > 0:
            trans_seq = [numpy.random.choice(list(self.transitions['#'].keys()), p=list(self.transitions['#'].values()))] * n
            em_seq = [numpy.random.choice(list(self.emissions[trans_seq[0]].keys()), p=list(self.emissions[trans_seq[0]].values()))] * n
            for i in range(1, n):
                next_state = trans_seq[i - 1]
                trans_seq[i] = numpy.random.choice(list(self.transitions[next_state].keys()), p=list(self.transitions[next_state].values()))
                next_state = trans_seq[i]
                em_seq[i] = numpy.random.choice(list(self.emissions[next_state].keys()), p=list(self.emissions[next_state].values()))
        else:
            print("Warning: You have generated an empty sequence!")
        return Sequence(trans_seq, em_seq)
    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

def main():
    h = HMM()
    if len(sys.argv) < 2:
        sys.exit("Usage: python HMM.py file_base [--generate num]")
    h.load(sys.argv[1])
    if len(sys.argv) > 3 and sys.argv[2] == '--generate':
        print(h.generate(int(sys.argv[3])))

if __name__ == "__main__":
    main()




