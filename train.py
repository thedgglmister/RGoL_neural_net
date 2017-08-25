# Standard library
import sys
import pickle

# Third-party libraries
from printcolors import PrintColors as Colors 
from network import RgolNetwork
from data_loader import TrainingLoader

argc = len(sys.argv)
if not 3 <= argc <= 4:
    usage = "Usage: train.py training_data_file dump_to_file [load_from_file]"
    sys.exit(Colors.RED + usage + Colors.WHITE)
training_data_file = sys.argv[1]
dump_to_file = sys.argv[2]
load_from_file = sys.argv[3] if argc == 4 else None

training_loader = TrainingLoader(training_data_file)
all_training_data = training_loader.load(fro=1, to=50000)
training_data = all_training_data[:40000]
validation_data = all_training_data[30000:40000]
evaluation_data = all_training_data[40000:50000]

if load_from_file:
	with open(load_from_file, "rb") as f:
		print(Colors.BLUE + '==> ' + Colors.CYAN + 'Loading network from "' + 
				load_from_file + '"')
		print(Colors.WHITE, end='')
		net = pickle.load(f)
else:
    net = RgolNetwork([405, 1, 400])
net.SGD(training_data,
	epochs=1,
	batch_size=10,
	eta=0.01,
	lmbda=5,
	validation_data=validation_data,
	evaluation_data=evaluation_data)
net.save(dump_to_file)
