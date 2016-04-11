import theano.tensor as T

import dlm.backend.nn_wrapper as K

class NegLogLikelihood():

	def __init__(self, classifier, args):

                self.y = K.placeholder(ndim=1, dtype='int32', name='y')
                self.w = K.placeholder(ndim=1, name='w')

		if args.instance_weights_path:
			self.cost = classifier.negative_log_likelihood(self.y, self.w)
		else:
			self.cost = classifier.negative_log_likelihood(self.y)

		if args.L1_reg > 0:
			self.cost = self.cost + args.L1_reg * classifier.L1

		if args.L2_reg > 0:
			self.cost = self.cost + args.L2_reg * classifier.L2_sqr

		if args.alpha and args.alpha > 0:
			self.cost = self.cost + args.alpha  * classifier.log_Z_sqr

		self.test = (
			K.mean(classifier.p_y_given_x(self.y))
		)
