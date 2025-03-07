# coding=utf-8

"""
Implementation of the RNN model
"""

import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf

from .utils import Variable, NLLLoss
from .vocabulary import Vocabulary
from IPython import embed

import tensorflow.compat.v1 as tf

import sys
import gym_molecule.models as gm_models
import gym_molecule.models.reinvent as gm_reinvent
import gym_molecule.models.reinvent.model as gm_model
import gym_molecule.models.reinvent.utils as gm_utils
import gym_molecule.models.reinvent.vocabulary as gm_vocab
# Trick Python into thinking 'models' is a top-level package
sys.modules["models"] = gm_models
sys.modules["models.reinvent"] = gm_reinvent
sys.modules["models.reinvent.model"] = gm_model
sys.modules["models.reinvent.utils"] = gm_utils
sys.modules["models.reinvent.vocabulary"] = gm_vocab


class MultiGRU(tnn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, gru_layer_size=512, num_gru_layers=3, embedding_layer_size=256):
        """
        Implements a N layer GRU(M) cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param gru_layer_size: Size of each of the GRU layers.
        :param num_gru_layers: Number of GRU layers.
        :param embedding_layer_size: Size of the embedding layer.
        """

        super(MultiGRU, self).__init__()

        self._gru_layer_size = gru_layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_gru_layers = num_gru_layers

        self._other_layers = {"embedding": tnn.Embedding(
            voc_size, self._embedding_layer_size)}
        self.add_module("embedding", self._other_layers["embedding"])

        layer = tnn.GRUCell(self._embedding_layer_size, self._gru_layer_size)
        self.add_module("gru_1", layer)
        self._gru_layers = [layer]
        for i in range(2, self._num_gru_layers + 1):
            layer = tnn.GRUCell(self._gru_layer_size, self._gru_layer_size)
            self._gru_layers.append(layer)
            self.add_module("gru_{}".format(i), layer)

        self._other_layers["linear"] = tnn.Linear(
            self._gru_layer_size, voc_size)
        self.add_module("linear", self._other_layers["linear"])

    def forward(self, *input_params):
        """
        Performs a forward pass on the model.
        :param x: Input tensor.
        :param h: Hidden state tensor.
        """
        input_vector = self._other_layers["embedding"](input_params[0])
        hidden_state = input_params[1]
        if hidden_state is None:
            hidden_state = Variable(torch.zeros(
                self._num_gru_layers, input_vector.size()[0], self._gru_layer_size))

        hidden_state_out = Variable(torch.zeros(hidden_state.size()))
        for i, gru_layer in enumerate(self._gru_layers):
            # print("\n\n input vector")
            # print(input_vector.size())
            # print(hidden_state.size())
            # print("\n\n hidden state")
            input_vector = hidden_state_out[i] = gru_layer(
                input_vector, hidden_state[i].cpu())

        input_vector = self._other_layers["linear"](input_vector)
        return input_vector, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'gru_layer_size': self._gru_layer_size,
            'num_gru_layers': self._num_gru_layers,
            'embedding_layer_size': self._embedding_layer_size
        }


class Model:
    """
    Implements an RNN model using SMILES.
    """

    def __init__(self, voc: Vocabulary, initialweights: OrderedDict = None, rnn_params=None):
        """
        Implements an RNN.
        :param voc: Vocabulary to use
        :param initialweights: Weights to initialize the RNN
        :param rnn_params: A dict with any of the accepted params in MultiGRU's constructor except for voc_size.
        """
        self.voc = voc

        if not isinstance(rnn_params, dict):
            rnn_params = {}

        self.rnn = MultiGRU(self.voc.vocab_size, **rnn_params)

        #if torch.cuda.is_available():
            #self.rnn.cuda()
        if initialweights:
            self.initialweights = copy.deepcopy(initialweights)
            self.rnn.load_state_dict(copy.deepcopy(initialweights))
        else:
            self.initialweights = copy.deepcopy(self.rnn.state_dict())

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Loads a model from a single file
        :param file: filpath as string
        :return: new instance of the RNN or None if it was not possible to load
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(
                file_path, map_location=lambda storage, loc: storage)
        model = Model(save_dict['vocabulary'],
                      initialweights=save_dict['initialweights'], rnn_params=save_dict.get("rnn_params", {}))
        model.rnn.load_state_dict(save_dict["currentweights"])
        return model

    def save(self, file):
        """
        Saves the model into a file
        :param file: Filepath as string
        """
        save_dict = {
            'vocabulary': self.voc,
            'initialweights': self.initialweights,
            'currentweights': self.rnn.state_dict(),
            'rnn_params': self.rnn.get_params()
        }
        torch.save(save_dict, file)

    def checkpoint(self):
        """
        Set self.initalweights to the current weights of self.rnn
        """
        self.initialweights = copy.deepcopy(self.rnn.state_dict())

    '''def likelihood(self, input_string, temperature=1.0):
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param target: (batch_size * sequence_length) A batch of sequences
        :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on
                            each position, but also more conservative.
                            Large values result in random predictions at each step.
        :return:  (batch_size) Log likelihood for each example.
        """
        # Convert string to tensor using vocabulary mapping
        #tokenized = self.voc.tokenize(smiles=input_string)
        #print(tokenized)
        vocab = {token: idx for idx, token in enumerate(sorted(set(tokenized)))}
        token_indices = [vocab[token] for token in tokenized]
        target = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0) 
        #print(target)
        batch_size, seq_length = target.size()
        #print(self.voc.vocab["^"])
        start_token = torch.tensor([[self.voc.vocab["^"]]], dtype=torch.long)
        input_vector = torch.cat((start_token, target[:, :-1]), dim=1)
        hidden_state = None
        #unfinished = torch.ones(1, 1, dtype=torch.uint8)
        log_prob_sum = 0.0
        
        try:
            #i =0
            for step in range(seq_length):
                #if i = 2:
                    #break
                print("+"*50)
                print("Corresponding token:", token_str)
                print("+"*50)
                logits, hidden_state = self.rnn(input_vector[:, step], hidden_state)
                logits = logits / temperature
                log_prob = tnnf.log_softmax(logits, dim=1)
                log_prob_step = log_prob[0, target[0, step]]  # Get log-probability of the actual target character
                log_prob_sum += log_prob_step.item()

                eos_sampled = (input_vector[:, step] == self.voc.vocab['$']).unsqueeze(1)
                unfinished = torch.eq(unfinished - eos_sampled, 1)
                if torch.sum(unfinished) == 0:
                    break
                i+=1
        
        except Exception as e:
            print("*"*50)
            print(e)
            print("*"*50)

        return log_prob_sum'''

    def likelihood(self, input_string, temperature=1.0):
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param target: (batch_size * sequence_length) A batch of sequences
        :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on
                            each position, but also more conservative.
                            Large values result in random predictions at each step.
        :return:  (batch_size) Log likelihood for each example.
        """
        batch_size = 1
        input_string = "^" + input_string
        input_string = self.voc.tokenize(smiles=input_string)
        #print(input_string)
        seq_length = len(input_string)
        # start_token = Variable(torch.zeros(batch_size).long())
        # start_token[:] = self.voc.vocab["^"]
        # start_token = start_token.cpu()
        hidden_state = None
        #input_vector = start_token
        #sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        #print(self.voc.vocab)
        pen = 0
        '''print("+"*50)
        print(input_vector, type(input_vector))
        print("+"*50)'''
        
        try:
            for step in range(seq_length):
                if input_string[step]=="$": 
                    break
                if input_string[step+1] not in self.voc.vocab.keys():
                    pen = 1000
                    break
                new_token_value = self.voc.vocab[input_string[step]]
                #print(new_token_value)
                new_token_tensor = torch.tensor([new_token_value], dtype=torch.long)
                #print(new_token_tensor.size())
                logits, hidden_state = self.rnn(new_token_tensor.cpu(), hidden_state)
                #print(logits.size(), hidden_state.size())
                logits = logits / temperature
                log_prob = tnnf.log_softmax(logits, dim=1)
                inc_token = torch.tensor(self.voc.vocab[input_string[step+1]])
                #log_prob_step = log_prob[0, input_string[0, step+1]]  # Get log-probability of the actual target character
                #log_prob_sum += log_prob_step
                #log_probs += log_prob.item()
                
                log_probs += NLLLoss(log_prob, inc_token)
                #print(pen)
        
        except Exception as e:
            print("*"*50)
            print(e)
            #print(input_string)
            print("*"*50)

        return log_probs + pen

    def sample_smiles(self, num=128, batch_size=128, temperature=1.0, sequence_length=140):
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on
                             each position, but also more conservative. Large values result in random predictions at
                             each step.
        :param sequence_lenght: Max number of tokens to sample per SMILES.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        batch_sizes = [batch_size for _ in range(
            num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        logging.debug("Sampling %d SMILES from Model", num)

        for size in batch_sizes:
            if not size:
                break
            seqs, likelihoods = self._sample(
                size, temperature=temperature, sequence_length=sequence_length)
            smiles = [self.voc.decode(seq) for seq in seqs.cpu().numpy()]

            smiles_sampled.extend(smiles)

            likelihoods_sampled.extend(likelihoods.data.cpu().numpy().tolist())
            del seqs, likelihoods

        return (smiles_sampled, likelihoods_sampled)

    def _sample(self, batch_size, sequence_length=140, temperature=1.0):
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab["^"]
        hidden_state = None
        input_vector = start_token
        unfinished = torch.ones_like(start_token, dtype=torch.uint8)
        sequences = []
        log_probs = Variable(torch.zeros(batch_size))

        for _ in range(sequence_length):
            logits, hidden_state = self.rnn(input_vector, hidden_state)
            logits = logits / temperature
            prob = tnnf.softmax(logits, dim=1)
            log_prob = tnnf.log_softmax(logits, dim=1)
            input_vector = torch.multinomial(prob, 1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            log_prob = log_prob * unfinished.unsqueeze(1).float()
            log_probs += NLLLoss(log_prob, input_vector)

            input_vector = Variable(input_vector.data)
            eos_sampled = (input_vector == self.voc.vocab['$'])
            unfinished = torch.eq(unfinished - eos_sampled, 1)
            if torch.sum(unfinished) == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs
