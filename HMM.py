

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
        matrix = [{} for i in range(len(sequence) + 1)]
        for key, _ in self.transitions.items():
            if key != "#":
                matrix[0][key] = 0.0
        # Deal with the starting state probability: #
        if len(sequence) > 0:
            for state in matrix[0]:
                if state not in self.transitions["#"] or sequence.outputseq[0] not in self.emissions[state]:   # If we can't transition to this state or emit this observation
                    matrix[1][state] = 0
                else:
                    matrix[1][state] = self.emissions[state][sequence.outputseq[0]]
                    matrix[1][state] *= self.transitions["#"][state]

        for i in range(1, len(sequence)):
            obsv = sequence.outputseq[i]
            for state in matrix[i]:
                matrix[i+1][state] = 0
                for state_prob in matrix[i]:
                    if state not in self.transitions[state_prob] or obsv not in self.emissions[state]:
                        prob = 0
                    else:
                        prob = self.emissions[state][obsv] # Emission probability
                        prob *= self.transitions[state_prob][state] # Multiplied by the transition probability
                        prob *= matrix[i][state_prob]
                    matrix[i+1][state] += prob
        # Finally: Get the state associated with the maximum value in the last dictionary in matrix
        max_state = ""
        max_prob = 0
        for state in matrix[-1]:
            if matrix[-1][state] > max_prob:
                max_prob = matrix[-1][state]
                max_state = state

        return state

    def viterbi(self, sequence):
        matrix = [{} for i in range(len(sequence) + 1)]
        for key, _ in self.transitions.items():
            if key != "#":
                matrix[0][key] = 0.0
        # Deal with the starting state probability: #
        if len(sequence) > 0:
            for state in matrix[0]:
                if state not in self.transitions["#"] or sequence.outputseq[0] not in self.emissions[state]:   # If we can't transition to this state or emit this observation
                    matrix[1][state] = 0
                else:
                    matrix[1][state] = self.emissions[state][sequence.outputseq[0]]
                    matrix[1][state] *= self.transitions["#"][state]

        for i in range(1, len(sequence)):
            obsv = sequence.outputseq[i]
            for state in matrix[i]:
                matrix[i+1][state] = 0
                for state_prob in matrix[i]:
                    if state not in self.transitions[state_prob] or obsv not in self.emissions[state]:
                        prob = 0
                    else:
                        prob = self.emissions[state][obsv] # Emission probability
                        prob *= self.transitions[state_prob][state] # Multiplied by the transition probability
                        prob *= matrix[i][state_prob]
                    matrix[i+1][state] += prob
        likely_states = [""] * len(matrix)
        index = 0
        for item in matrix:
            max_prob = 0
            max_state = ""
            for state in item:
                if item[state] > max_prob:
                    max_prob = item[state]
                    max_state = state
            likely_states[index] = max_state
            index += 1
        return likely_states
        # Deal with the starting state probability: #
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

    def write_obs_files(self, base):
        """
        Write code to generate length 20 sequences for cat and lander
        Write those sequences to cat_sequence.obs and lander_sequence.obs
        :return: None
        """
        obsv = self.generate(20)
        with open(base + "_sequence.obs", "w") as f: f.write(obsv.__str__())


def parse_obs(h, obs_file, algo):
    try:
        with open(obs_file) as f: observations = f.read()
    except FileNotFoundError:
        print(f"Warning: {obs_file} does not exist")
        observations = []
    observations = observations.split('\n')
    for i in range(1, len(observations), 2):
        line = observations[i]
        print("For observations:", line)
        print("Predicted state: ", end="")
        if obs_file.startswith("lander"):
            if algo == "forward":
                final_spot = h.forward(Sequence(stateseq=[], outputseq=line.split(" ")))
                if final_spot in ["2,5", "3,4", "4,3", "4,4", "5,5"]:
                    print("Safe to land!")
                else:
                    print("Not safe to land! (", final_spot, ")", sep="")
            else:
                print(h.viterbi(Sequence(stateseq=[], outputseq=line.split(" "))))
        else:
            if algo == "forward":
                print(h.forward(Sequence(stateseq=[], outputseq=line.split(" "))))
            else:
                print(h.viterbi(Sequence(stateseq=[], outputseq=line.split(" "))))

def main():
    h = HMM()
    parse = argparse.ArgumentParser()
    parse.add_argument("file", help=".trans and .emit base name")
    parse.add_argument("--generate", help="--generate number")
    parse.add_argument("--forward", help="--forward .obs_file_name")
    parse.add_argument("--viterbi", help="--viterbi .obs_file_name")
    args = parse.parse_args()
    h.load(args.file)
    if args.generate:
        print(h.generate(int(args.generate)))
    if args.forward:
        if args.forward.startswith("ambiguous"):
            parse_obs(h, "ambiguous_sents.obs", "forward")
        else:
            parse_obs(h, args.forward, "forward")
    if args.viterbi:
        if args.viterbi.startswith("ambiguous"):
            parse_obs(h, "ambiguous_sents.obs", "viterbi")
        else:
            parse_obs(h, args.viterbi, "viterbi")

if __name__ == "__main__":
    main()




