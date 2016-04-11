#import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import math
import dlm.backend.nn_wrapper as K


class NCELikelihood():

	def __init__(self, classifier, args, noise_dist):
                self.y = K.placeholder(ndim=1, name='y')

		## Cost function
		#  Sum over minibatch instances (log ( u(w|c) / (u(w|c) + k * p_n(w)) ) + sum over noise samples ( log ( u(x|c) / ( u(x|c) + k * p_n(x) ) )))

		# Generating noise samples
		srng = RandomStreams(seed=1234)
		noise_samples = srng.choice(size=(self.y.shape[0],args.num_noise_samples),  a=args.num_classes, p=noise_dist, dtype='int32')
                log_noise_dist = K.variable(np.log(noise_dist.get_value())
		#log_noise_dist = theano.shared(np.log(noise_dist.get_value()),borrow=True)
		#log_num_noise_samples = theano.shared(math.log(args.num_noise_samples)).astype(theano.config.floatX)
                #TODO: Are we sure this is K.variable?
		log_num_noise_samples = K.variable(np.log(args.num_noise_samples,dtype=K._FLOATX))
		# Data Part of Cost Function: log ( u(w|c) / (u(w|c) + k * p_n(w))
		data_scores = classifier.output[K.arange(self.y.shape[0]),self.y]
		data_denom = self.logadd(data_scores, log_num_noise_samples + log_noise_dist[self.y] )
		data_prob = data_scores - data_denom
		# Sumation of Noise Part of Cost Function: sum over noise samples ( log ( u(x|c) / ( u(x|c) + k * p_n(x) ) ))
		noise_mass = log_num_noise_samples + log_noise_dist[noise_samples] # log(k) + log(p_n(x)) for all noise samples (Shape: #instaces x k)
		noise_scores = classifier.output[K.arange(noise_samples.shape[0]).reshape((-1,1)),noise_samples]
		noise_denom = self.logadd(noise_scores, noise_mass)
		noise_prob_sum = K.sum(noise_mass - noise_denom, axis=1)

		self.cost = (
			-K.mean(data_prob + noise_prob_sum)
		)
		self.test = (
			K.sum(data_scores)
		)

	def logadd(self, a, b):
		g = K.maximum(a,b)
		l = K.minimum(a,b)
		return g + K.log(1 + K.exp(l-g))

