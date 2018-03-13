########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        ### TODO: Insert Your Code Here (2A)
        #probs[0] = np.log(self.A_start)
        probs[0] = self.A_start
        probs[1] = np.multiply(probs[0], [self.O[i][x[0]] for i in range(self.L)])
        seqs[1] = [str(i) for i in range(self.L)] 
        for i in np.arange(1, M):
            for j in range(self.L):
                p = []
                for l in range(self.L):
                    p_temp = probs[i][l]*self.A[l][j]*self.O[j][x[i]]
                    p.append(p_temp)
                probs[i+1][j] = max(p)
                seqs[i+1][j] = seqs[i][p.index(max(p))] + str(j)
        
        p_final = probs[M]
        max_seq = seqs[M][p_final.index(max(p_final))]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ### TODO: Insert Your Code Here (2Bi)
        alphas[1] = np.multiply([1. / self.L for _ in range(self.L)], [self.O[i][x[0]] for i in range(self.L)])
        for i in np.arange(2, M + 1):
            for j in range(self.L):
                sum = 0
                for k in range(self.L):
                    sum += alphas[i - 1][k]*self.A[k][j]
                alphas[i][j] = self.O[j][x[i - 1]]*sum
            if normalize:
                norm = np.sum(alphas[i])
                for j in range(self.L):
                    alphas[i][j] /= norm
                
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ### TODO: Insert Your Code Here (2Bii)
        betas[M] = [1. for _ in range(self.L)]
        for index in range(M):
            i = M - 1 - index
            for j in range(self.L):
                sum = 0
                for k in range(self.L):
                    sum += betas[i + 1][k]*self.A[j][k]*self.O[k][x[i]]
                betas[i][j] = sum
            if normalize:
                norm = np.sum(betas[i])
                for j in range(self.L):
                    betas[i][j] /= norm
                
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        ### TODO: Insert Your Code Here (2C)
        init_A = np.copy(self.A)
        for seq in Y:
            N = len(seq)
            for i in range(N - 1):
                self.A[seq[i]][seq[i + 1]] += 1
        
        self.A = np.subtract(self.A, init_A)
        for i in range(len(self.A)):
            norm = sum(self.A[i])
            for j in range(len(self.A[i])):
                self.A[i][j] /= norm
        
        # Calculate each element of O using the M-step formulas.
        ### TODO: Insert Your Code Here (2C)
        init_O = np.copy(self.O)
        for x, y in zip(X, Y):
            N = len(x)
            for i in range(N):
                self.O[y[i]][x[i]] += 1
                
        self.O = np.subtract(self.O, init_O)
        for i in range(len(self.O)):
            norm = sum(self.O[i])
            for j in range(len(self.O[i])):
                self.O[i][j] /= norm

        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ### TODO: Insert Your Code Here (2D)
        for i in range(N_iters):
            print(i)
            P1 = []
            P2 = []
            # E step:
            for seq in X:
                M = len(seq)
                alphas = self.forward(seq, normalize=True)
                betas = self.backward(seq, normalize=True)
                p1 = [[0. for _ in range(self.L)] for _ in range(M)]
                p2 = [np.zeros([self.L, self.L]) for _ in range(M - 1)]
                for j in range(M):
                    # To calculate P(y^j=a,x) (P1)
                    for k in range(self.L):
                        p1[j][k] += alphas[j + 1][k] * betas[j + 1][k]
                    norm = sum(p1[j])
                    for k in range(self.L):
                        p1[j][k] /= norm
                for j in range(M - 1):
                    # To calculate P(y^j=a,y^j+1=b,x) (P2)
                    for k1 in range(self.L):
                        for k2 in range(self.L):
                            p2[j][k1][k2] += alphas[j + 1][k1] * self.O[k2][seq[j + 1]] * self.A[k1][k2] * betas[j + 2][k2]
                    summing = sum(sum(p2[j]))
                    for k1 in range(self.L):
                        for k2 in range(self.L):
                            p2[j][k1][k2] /= summing
                P1.append(p1)
                P2.append(p2)
            
            # M step:
            init_A = np.copy(self.A)
            init_O = np.copy(self.O)
            self.A = np.subtract(self.A, init_A)
            self.O = np.subtract(self.O, init_O)
            for seq, p1 in zip(X, P1):
                M = len(seq)
                for i in range(M):
                    for j in range(self.L):
                        self.O[j][seq[i]] += p1[i][j]
            for i in range(len(self.O)):
                norm = sum(self.O[i])
                for j in range(len(self.O[i])):
                    self.O[i][j] /= norm
            
            for p2 in P2:
                for mat in p2:
                    self.A = np.add(self.A, mat)
            for i in range(len(self.A)):
                norm = sum(self.A[i])
                for j in range(len(self.A[i])):
                    self.A[i][j] /= norm
            
        pass


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        ### TODO: Insert Your Code Here (2F)
        init_state = np.random.choice(self.L)
        init_emission = np.random.choice(self.D, p=self.O[init_state])
        states.append(init_state)
        emission.append(init_emission)
        for i in np.arange(1, M):
            s = np.random.choice(self.L, p=self.A[states[i - 1]])
            e = np.random.choice(self.D, p=self.O[s])
            states.append(s)
            emission.append(e)
        
        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(420)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]
    
    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    random.seed(69)
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    print('start!')
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
