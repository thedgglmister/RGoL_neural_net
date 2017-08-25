"""This module contains the CrossEntropyCost, DataSet, Network,
and RgolNetwork classes."""

# Standard Library
import random
import time
import pickle
import csv

# Third-party libraries
import numpy as np
from printcolors import PrintColors as Colors

class CrossEntropyCost:
	"""This class represents a function used to assess the error of a neural
	network. It is supplied as an argument when initializing a Network object.

	Methods include fn() and delta().

	fn(a, y) is the actual cost function. a and y are both n x 1 2D numpy 
	arrays, with a representing the activation levels of the output layer in a
	neural network and y representing the expected values.

	delta(a, y) is the derivative of the cross-entropy cost function.
	a and y are both n x 1 2D numpy arrays, with a representing the activation 
	levels of the output layer in a neural network and y representing the 
	expected values."""

	@staticmethod
	def fn(a, y):
		"""The cross-entropy cost function. a and y are both n x 1 2D numpy
		arrays, with a representing the activation levels of the output layer
		in a neural network and y representing the expected values"""
		return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

	@staticmethod
	def delta(a, y):
		"""the derivative of the cross-entropy cost function. a and y are 
		both n x 1 2D numpy arrays, with a representing the activation levels 
		of the output layer in a neural network and y representing the 
		expected values."""
		return a - y


class DataSet:
	"""The DataSet class contains information about subsets of training data
	used while training a Network object.

	It is initialized with a 2D numpy array of data, as well as a string 
	representing the n	ame of the data set"""

	def	__init__(self, data, name):
		"""Initializes object with data, cnt, name, cost_log, and accuracy_log.
		data is a 2D numpy array of data, cnt contains teh length of the data 
		set, name is a string representing the name of the data set. cost_log 
		and accuracy log are lists to be populated each epoch of training if 
		cost and accuracy are monitored."""
		self.data = data
		self.cnt = len(data)
		self.name = name
		self.cost_log = []
		self.accuracy_log = []


class Network:
	"""A class representing a feed forward neural network.

	A Network object is initialized with a list of positive integers
	representing the layers of the network. For example, [20, 10, 2] represents 
	a network with 20 input neurons, a single hidden layer with 10 neurons, and 
	2 output neurons. At initialization a cost object is also supplied, which 
	should include both a fn() and delta() method.

	public methods include SGD(), save(), and predict().

	SGD(self, training_data, epochs=1, batch_size=10, eta=1, lmbda=0.0,
			validation_data=None, evaluation_data=None) implements a stochastic 
	gradient descent algorithm used to train the Network object.
	training_data is the data used to train the Network and should be a list of 
	tuples of the form 
	([input1, input2, ...], [expected_output1, expected_output2, ...]).
	epochs is the number of epochs the Network will train for.
	batch_size is the number of data pieces fed into back propogation at a time.
	eta is the learning rate.
	lmbda is the regularization coefficient used to prevent overfitting.
	validation_data and evaluation_data can be supplied in order to assess 
	training progress. If supplied, the cost and accuracy will be logged after 
	each epoch.

	save(file_name) will pickle the Network object and save it to file_name.

	predict(input_data) is used to run input_data into the Network and save the
	outputs to a file.
	
	IMPORTANT: since measuring accuracy and interpreting output activations 
	depends on the specific problem being solved by a Network, the predict(), 
	accuracy(), and log_accuracy() are left unimplemented in the Network class 
	and must be implemented in a subclass."""

	def __init__(self, layer_sizes, cost=CrossEntropyCost):
		"""Initializes Network with a list of positive integers representing 
		the layers of the network. For example, [20, 10, 2] represents a network
		with 3 layers of sizes 20, 10, and 2 neurons each. At initialization a 
		cost object is also supplied, which should include both a fn() and 
		delta() method. The Network is also randomly initialized with biases and 
		weights."""
		self.layer_cnt = len(layer_sizes)
		self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
		self.weights = [np.random.randn(y, x) / np.sqrt(x) 
				for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
		self.cost = cost

	@staticmethod
	def _sigmoid(z):
		"""Accepts 2D numpy array of weighted inputs for a network layer as
		argument and returns a 2D numpy array contaiing the layer's activation 
		levels."""
		return 1 / (1 + np.exp(-z))

	def _sigmoid_prime(self, z):
		"""derivative of sigmoid function"""
		return self._sigmoid(z) * (1 - self._sigmoid(z))

	def _feed_forward(self, a):
		"""Accepts a 2D numpy array of inputs to the network and returns a 2D 
		numpy array representing the activation levels of the output layer."""
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, a) + b
			a = self._sigmoid(z)
		return a

	def _back_prop(self, inpt, goal):
		"""Implements the back propogation algorithm to adjust weights and 
		biases of the Network to decrease cost. Accepts 2D numpy arrays of input
		values and expected outputs and returns numpy matrices the same size as 
		the Network's weights and biases populated with deltas to be applied."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = inpt
		activations = [inpt]
		weighted_inpts = []
		for b, w in zip(self.biases, self.weights):
			weighted_inpt = np.dot(w, activation) + b
			weighted_inpts.append(weighted_inpt)
			activation = self._sigmoid(weighted_inpt)
			activations.append(activation)
		delta = self.cost.delta(activations[-1], goal)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for layer in range(2, self.layer_cnt):
			weighted_inpt = weighted_inpts[-layer]
			sp = self._sigmoid_prime(weighted_inpt)
			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
			nabla_b[-layer] = delta
			nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
		return nabla_b, nabla_w

	def _update_network(self, batch, eta, lmbda, training_cnt):
		"""Feeds the training tuples in batch to back_prop() and then uses the
		result to adjust the weights and biases of the Network. eta is the 
		learning rate, lmbda is the regularization coefficient, and training_cnt 
		is the total size of the training_data."""
		batch_cnt = len(batch)
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for inpt, goal in batch:
			delta_nabla_b, delta_nabla_w = self._back_prop(inpt, goal)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [(1 - eta * lmbda/training_cnt) * w - eta * nw/batch_cnt
				for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - eta * (nb / batch_cnt)
				for b, nb in zip(self.biases, nabla_b)]

	def _total_cost(self, data, lmbda):
		"""Computes the total cost of a training set. lmbda is the 
		regularization coefficient and data is a list of tuples, where each tuple 
		is of the form 
		([input1, input2, ...], [expected_output1, expected_output2, ...])."""
		cost = 0.0
		for inpt, goal in data:
			a = self._feed_forward(inpt)
			cost += self.cost.fn(a, goal) / len(data)
		cost += (0.5 * (lmbda/len(data)) *
				sum(np.linalg.norm(w)**2 for w in self.weights))
		return cost

	def _log_cost(self, data_set, lmbda):
		"""Calculates the total cost of a data set,records the result in the
		cost_log of the DataSet object, and prints the result. data_set is a 
		DataSet object and lmbda is the regularization coefficient."""
		cost = self._total_cost(data_set.data, lmbda)
		data_set.cost_log.append(cost)
		print(Colors.YELLOW, end="")
		print("        Cost on {0} data: {1:.5f}".format(data_set.name, cost))
		print(Colors.WHITE, end='')

	def _accuracy(self, data):
		""""""
		raise NotImplementedError("Must be implemented by subclass")


	def _log_accuracy(self, data_set):
		""""""
		raise NotImplementedError("Must be implemented by subclass")
	
	def _display_logs(self, 
			  		  lmbda,
			  		  validate_set,
			  		  evaluate_set):
		"""If validation and evaluation data sets are provided to SGD(), this
		method calculates, logs, and prints the cost and accuracy of each data 
		set after each epoch of trianing."""
		if validate_set:
			self._log_cost(validate_set, lmbda)
		if evaluate_set:
			self._log_cost(evaluate_set, lmbda)
		if validate_set:
			self._log_accuracy(validate_set)
		if evaluate_set:
			self._log_accuracy(evaluate_set)

	@staticmethod
	def _consolidate_logs(validate_set, evaluate_set):
		"""Consolidates the cost and accuracy logs of the validation and 
		evaluation data sets into a hash table."""
		consolidated = {}
		if validate_set:
			consolidated["validation_cost"] = validate_set.cost_log
			consolidated["validation_accuracy"] = validate_set.accuracy_log
		if evaluate_set:
			consolidated["evaluation_cost"] = evaluate_set.cost_log
			consolidated["evaluation_accuracy"] = evaluate_set.accuracy_log
		return consolidated

	def SGD(self, 
			training_data, 
			epochs=1, 
			batch_size=10, 
			eta=1, 
			lmbda=0.0,
			validation_data=None,
			evaluation_data=None):
		"""Implements a stochastic gradient descent algorithm used to train the 
		Network object. training_data is the data used to train the Network and 
		should be a list of tuples of the form 
		([input1, input2, ...], [expected_output1, expected_output2, ...]).
		epochs is the number of epochs the Network will train for.
		batch_size is the number of data pieces fed into back propogation at a 
		time. eta is the learning rate.
		lmbda is the regularization coefficient used to prevent overfitting.
		validation_data and evaluation_data can be supplied in order to assess 
		training progress. If supplied, the cost and accuracy will be logged 
		after each epoch."""
		train_set = DataSet(training_data, "training")
		validate_set = (DataSet(validation_data, "validation") 
				if validation_data else None)
		evaluate_set = (DataSet(evaluation_data, "evaluation")
				if evaluation_data else None)
		total_time = 0
		print(Colors.BLUE + "==> " + Colors.CYAN + "Begin Training")
		print(Colors.WHITE, end='')
		for i in range(epochs):
			start_time = time.monotonic()
			random.shuffle(train_set.data)
			batches = [train_set.data[k : k + batch_size]
					for k in range(0, train_set.cnt, batch_size)]
			for batch in batches:
				self._update_network(batch, eta, lmbda, train_set.cnt)
			elapsed_time = time.monotonic() - start_time
			total_time += elapsed_time
			progress_msg = "Epoch {0} / {1} completed in {2:.3f} secs"
			progress_msg = progress_msg.format(i + 1, epochs, elapsed_time)
			print(Colors.BLUE + "==> " + Colors.CYAN + progress_msg)
			print(Colors.WHITE, end='')
			self._display_logs(lmbda, validate_set, evaluate_set)
		print(Colors.BLUE + "==> " + Colors.CYAN, end='')
		print("Training complete in {0:.4f} secs".format(total_time))
		print(Colors.WHITE, end='')
		return self._consolidate_logs(validate_set, evaluate_set)

	def predict(self, inpt_data):
		"""predict(input_data) is used to run input_data into the Network and
		save the outputs to a file. Since interpreting output activations
		depends on the specific problem being solved by a Network, this method 
		must be implemented in a subclass."""
		raise NotImplementedError("Must be implemented by subclass of Network")
	
	def save(self, file_name):
		"""Pickles the Network object and saves it to file_name"""
		with open(file_name, "wb") as f:
			pickle.dump(self, f)
		print(Colors.BLUE + '==> ' + Colors.CYAN, end='')
		print('Network saved to "{0}"'.format(file_name))
		print(Colors.WHITE, end='')
						

class RgolNetwork(Network):
	"""Subclass of Network that represents a neural network to be used in 
	solving Conway's Reverse Game of Life.

	Contains the methods accuracy(), log_accuracy(), and predict() that are left
	unimplemented in the Network class.

	accuracy(data) returns the number of correct predictions.

	log_accuracy(data_set) logs the accuracy in a DataSet's accuracy_log.

	predict(input_data, file_name) outputs predictions based on input_data to
	file_name."""

	def __init__(self, layer_sizes, cost=CrossEntropyCost):
		"""Initializes RgolNetwork with a list of positive integers representing 
		the layers of the network. For example, [20, 10, 2] represents a network
		with 3 layers of sizes 20, 10, and 2 neurons each. At initialization a 
		cost object is also supplied, which should include both a fn() and 
		delta() method. The RgolNetwork isalso randomly initialized with biases 
		and weights."""
		super().__init__(layer_sizes, cost)
	
	def _accuracy(self, data):
		"""Returns the total correct classification of cells as dead or alive. 
		data is a list of tuples, where each tuple is of the form 
		([input1, input2, ...], [expected_output1, expected_output2, ...])."""
		total_correct = 0
		for inpt, goal in data:
			result = [round(x[0]) for x in self._feed_forward(inpt)]
			for x, y in zip(result, goal):
				if x == y[0]:
					total_correct += 1
		return total_correct

	def _log_accuracy(self, data_set):
		"""Calculates the accuracy percentage of a data set, records the result
		in the accuracy_log of the DataSet object, and prints the result. 
		data_set is a DataSet object."""
		total = self._accuracy(data_set.data)
		accuracy = float(total) / (data_set.cnt * 400)
		data_set.accuracy_log.append(accuracy)
		print(Colors.YELLOW + "        " +
				"Accuracy on {0} data: {1:.5f}".format(data_set.name, accuracy))
		print(Colors.WHITE, end='')
	
	def predict(self, inpt_data, file_name):
		"""Runs input_data into the Network and save the outputs to a file_name. 
		imput_data is a 2D numpy array and file_name is the name of the file
		to save the results to."""
		with open(file_name, "w") as f:
			csv_writer = csv.writer(f)
			header = ["id"]
			for i in range(1, 401):
				header.append("start." + str(i))
			print(Colors.BLUE + "==> " + Colors.CYAN + "Begin testing")
			print(Colors.WHITE, end='')
			csv_writer.writerow(header)
			for i, inpt in enumerate(inpt_data):
				prediction = [i + 1]
				prediction.extend([int(round(x[0])) 
						for x in self._feed_forward(inpt)])
				csv_writer.writerow(prediction)
		print(Colors.BLUE + '==> ' + Colors.CYAN, end='')
		print('Results saved to "' + file_name + '"')
		print(Colors.WHITE, end='')
