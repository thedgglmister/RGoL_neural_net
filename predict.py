# Standard Library
import pickle
import sys

# Third-party libraries
from printcolors import PrintColors as Colors
from data_loader import TestLoader

argc = len(sys.argv)
if argc != 4:
    usage = "Usage: predict.py test_data_file load_from_file output_file"
    sys.exit(Colors.RED + usage + Colors.WHITE)
test_data_file = sys.argv[1]
load_from_file = sys.argv[2]
output_file = sys.argv[3]

test_loader = TestLoader(test_data_file)
test_data = test_loader.load(fro=1)

with open(load_from_file, "rb") as f:
    print(Colors.BLUE + '==> ' + Colors.CYAN + 'Loading network from "' +
	    load_from_file + '"')
    print(Colors.WHITE, end='')
    net = pickle.load(f)
net.predict(test_data, output_file)
