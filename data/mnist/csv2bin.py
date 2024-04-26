
# This code is used to convert .csv files downloaded from 
# https://python-course.eu/data/mnist/mnist_train.csv and https://python-course.eu/data/mnist/mnist_test.csv
# to .ubyte binary files.

# Since Github rejects pushing files that are larger than 100MB, we needed to compress the MNIST data set.

import numpy as np

# Image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Load the train set
train_data = np.loadtxt("mnist_train.csv", delimiter=",", dtype=np.uint8)
train_bytes = []
for image in train_data:
	train_bytes.extend(image) 
train_bytes = bytearray(train_bytes)

# Load the test set
test_data = np.loadtxt("mnist_test.csv", delimiter=",", dtype=np.uint8) 
test_bytes = []
for image in test_data:
	test_bytes.extend(image) 
test_bytes = bytearray(test_bytes)

# Write the train data to a binary file
train_filename = "mnist_train.ubyte"
with open(train_filename, "wb") as binary_file:
	binary_file.write(train_bytes)

# Write the test data to a binary file
test_filename = "mnist_test.ubyte"
with open(test_filename, "wb") as binary_file:
	binary_file.write(test_bytes)
	
# Read the train data from the binary file, and check whether it matches with the original data
with open(train_filename, "rb") as binary_file:
    train_bytes = binary_file.read()
    byte_number = len(train_bytes)
    image_number = byte_number//(IMAGE_SIZE+1)

    if (image_number * (IMAGE_SIZE+1) != byte_number):
        raise Exception("Dimensions do not match in train set")
    
    train_data_read = np.ndarray(shape=(image_number, IMAGE_SIZE+1), dtype=np.uint8)
    for i in range(image_number):
        for j in range(IMAGE_SIZE+1):
            train_data_read[i][j] = train_bytes[i*(IMAGE_SIZE+1) + j]
            if (train_data_read[i][j] != train_data[i][j]):
                raise Exception("Train set has been written to %s incorrectly" % train_filename)
    print("Train set has been written to %s successfully" % train_filename)


# Read the test data from the binary file, and check whether it matches with the original data
with open(test_filename, "rb") as binary_file:
    test_bytes = binary_file.read()
    byte_number = len(test_bytes)
    image_number = byte_number//(IMAGE_SIZE+1)

    if (image_number * (IMAGE_SIZE+1) != byte_number):
        raise Exception("Dimensions do not match in test set")
    
    test_data_read = np.ndarray(shape=(image_number, IMAGE_SIZE+1), dtype=np.uint8)
    for i in range(image_number):
        for j in range(IMAGE_SIZE+1):
            test_data_read[i][j] = test_bytes[i*(IMAGE_SIZE+1) + j]
            if (test_data_read[i][j] != test_data[i][j]):
                raise Exception("Test set has been written to %s incorrectly" % test_filename)
    print("Test set has been written to %s successfully" % test_filename)


