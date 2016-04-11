import sys
#import theano
#import theano.tensor as T
import numpy
import dlm.utils as U
import dlm.io.logging as L
from dlm.io.vocabReader import VocabManager
from dlm.io.w2vEmbReader import W2VEmbReader
import dlm.backend.nn_wrapper as K

class LookupTable():

	def __init__(self, rng, input, vocab_size, emb_dim, emb_matrix=None, concat=True, emb_path=None, vocab_path=None, add_weights=False):

		L.info("Lookup Table layer, #words: %s, #dims: %s" % (U.red(vocab_size), U.red(emb_dim)))

		self.input = input
		L.info("Input " + str(input))
                L.info("Add weightes " + str(add_weights))
                self.emb_matrix = emb_matrix

		if self.emb_matrix is None:
			self.emb_matrix = numpy.asarray(
				rng.uniform(
					low=-0.01, #low=-1,
					high=0.01, #high=1,
					size=(vocab_size, emb_dim)
				),
				dtype=K._FLOATX
			)

		if emb_path:
			U.xassert(vocab_path, 'When emb_path is given, vocab must be given too.')
			self.initialize(emb_path, vocab_path)

		#self.embeddings = theano.shared(value=self.emb_matrix, name='embeddings', borrow=True)
		self.embeddings = K.variable(self.emb_matrix, name='embeddings')


		if add_weights:
			weights_vec = numpy.ones(vocab_size, dtype=K._FLOATX)
			#self.weights = theano.shared(value=weights_vec, name='word_weights', borrow=True)
			self.weights = K.variable(weights_vec, name='word_weights')

			# Check if the speed can be improved
			self.output = (self.weights.dimshuffle(0, 'x') * self.embeddings)[input]
			#self.output = self.weights.dimshuffle(0, 'x')[input] * self.embeddings[input]
			#self.output = self.weights[input].dimshuffle(0, 'x') * self.embeddings[input]

			self.params = [self.embeddings, self.weights]
		else:
			self.output = self.embeddings[input]
			self.params = [self.embeddings]

		if concat:
			self.output = self.output.reshape((input.shape[0], emb_dim * input.shape[1]))

	def initialize(self, emb_path, vocab_path):
		L.info('Initializing lookup table')
		vm = VocabManager(vocab_path)
		w2v = W2VEmbReader(emb_path)
		U.xassert(w2v.get_emb_dim() == self.emb_matrix.shape[1], 'The embeddings dimension does not match with the given word embeddings')
		for i in range(self.emb_matrix.shape[0]):
			vec = w2v.get_emb_given_word(vm.get_word_given_id(i))
			if vec:
				self.emb_matrix[i] = vec
