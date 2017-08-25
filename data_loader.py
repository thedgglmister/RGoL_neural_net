"""This module contains the DataLoader, TrainingData and TestData classes.

These subclasses are used to load data from a csv file into a list for use in
solving Conway's Reverse Game of Life."""

# Third-party libraries
import numpy as np
from printcolors import PrintColors as Colors


class DataLoader:
	"""DataLoader is a baseclass for loading data from a csv file."""

	def __init__(self, file_name):
		"""Initializes the loader with the name of the csv file."""
		self.file_name = file_name

	def load(self):
		"""An abstract method to be implemented in a subclass."""
		raise NotImplementedError("Must be implemented in subclass")


class TrainingLoader(DataLoader):
	"""Loads data from a csv file to be used to train a RgolNetwork.

	Methods include __init__() and load().

	__init__(file_name) initializes the loader with the name of the csv file.
	The lines of the csv should be in the form
	id, delta, start.1, ..., start.400, ..., stop.1, ..., stop.400

	load(fro=0, to=None) loads lines from the csv to tuples containing the input 
	and expected output for use in Training a RgolNetwork. The return is a list
	of such tuples. fro and to are used to specify an inclusive range of lines
	to be loaded."""

	def __init__(self, file_name):
		"""Initializes the loader with the name of the csv file."""
		self.file_name = file_name
		super().__init__(file_name)

	def load(self, fro=0, to=None):
		"""Loads lines from the csv to tuples of lists containing the input and 
		expected output for use in Training a RgolNetwork. The return is a list
		of such tuples. fro and to are used to specify an inclusive range of 
		lines to be loaded."""
		training_data = []
		print(Colors.BLUE + '==> ', end='')
		print(Colors.CYAN + 'Loading training data from "' + 
				self.file_name + '"')
		print(Colors.WHITE, end='')
		with open(self.file_name, "r") as f:
			lines = f.readlines()
			if not to:
				to = len(lines) - 1
			for line in lines[fro:to + 1]:
				line = line.strip().split(",")
				matrix = [[int(x)] for x in line]
				inpt = [[0], [0], [0], [0], [0]]
				delta = matrix[1][0]
				inpt[delta - 1][0] = 1
				inpt.extend(matrix[402:])
				outpt = matrix[2:402]
				training_pair = (np.array(inpt), np.array(outpt))
				training_data.append(training_pair)
		print(Colors.BLUE + '==> ' + Colors.CYAN + 'Loading complete')
		print(Colors.WHITE, end='')
		return training_data


class TestLoader(DataLoader):
	"""Loads data from csv file to be used by a RgolNetwork to make predictions.

	Methods include __init__() and load().

	__init__(file_name) initializes the loader with the name of the csv file.
	The lines of the csv should be in the form id, start.1, ..., start.400
 	
	load(fro=0, to=None) loads lines from the csv to lists containing the input 
	to be provided to a RgolNetwork. The return is a list of these lists.
	fro and to are used to specify an inclusive range of lines to be loaded."""

	def __init__(self, file_name):
		"""Initializes the loader with the name of the csv file."""
		super().__init__(file_name)

	def load(self, fro=0, to=None):
		"""loads lines from the csv to lists containing the input to be provided 
		to a RgolNetwork. The return is a list of these lists. fro and to are 
		used to specify an inclusive range of lines to be loaded."""
		test_data = []
		print(Colors.BLUE + '==> ', end='')
		print(Colors.CYAN + 'Loading testing data from "' + 
				self.file_name + '"')
		print(Colors.WHITE, end='')
		with open(self.file_name, "r") as f:
			lines = f.readlines()
			if not to:
				to = len(lines) - 1
			for line in lines[fro:to + 1]:
				line = line.strip().split(",")
				matrix = [[int(x)] for x in line]
				inpt = [[0], [0], [0], [0], [0]]
				delta = matrix[1][0]
				inpt[delta - 1][0] = 1
				inpt.extend(matrix[2:])
				test_data.append(np.array(inpt))
		print(Colors.BLUE + '==> ' + Colors.CYAN + 'Loading complete')
		print(Colors.WHITE, end='')
		return test_data
