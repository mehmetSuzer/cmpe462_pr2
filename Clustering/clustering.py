
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import *

# Image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Plots first row_number X col_number many images onto one grid
def plot_images(images: np.ndarray, row_number: int, col_number: int) -> None:
    image_number = len(images)
    if (image_number < row_number * col_number):
        raise Exception("argument 'images' have %d images, but the expected grid is %dX%d" % (image_number, row_number, col_number)) 
    
    _, axes = plt.subplots(row_number, col_number, figsize=(10,4))
    for i, ax in enumerate(axes.flat):
        img = images[i].reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.show()

def plot_line_graph(values: list, title: str, x_label: str, y_label: str) -> None:
    plt.plot(values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.legend()
    plt.show()

# Returns the magnitude of a vector
def magnitude(arr: np.ndarray) -> float:
    return np.dot(arr, arr) ** 0.5

# Returns the euclidean similarity of two vectors.
# The higher the distance between two vectors, the smaller the similarity.
def euclidean_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    mag = magnitude(arr1 - arr2)
    if (mag > 1E-6):
        return 1.0 / mag
    else:
        return 1E6

# Returns the cosine similarity of two vectors.
# The higher the angle between two vectors, the smaller the similarity.
def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    return np.dot(arr1, arr2) / (magnitude(arr1) * magnitude(arr2))

# Returns the Sum of Squared Error (SSE) of the model developed by the K-means clustering algorithm.
def sum_squared_error(train_x: np.ndarray, centroids: list[np.ndarray], membership: np.ndarray) -> float:
    error = 0.0
    for cluster in range(class_number):
        for i in range(train_size):
            if (membership[cluster][i] == 1):
                distance = magnitude(train_x[i] - centroids[cluster])
                error += distance
    return error

# Runs k-means clustering algorithms on 'x' with 'k' clusters by using 'similarity_function'.
# Returns 'k' centroids and the membership matrix.
# Prints how many points have changed their clusters for each cluster.
# When all clusters are subject to 'stop_threshold' many point changes, the algorithm stops.
def k_means_clustering(x: np.ndarray, k: int, similarity_function: Callable[[np.ndarray, np.ndarray], float], stop_threshold: int = 4) -> tuple[list, list, np.ndarray]:
    # generate random centroids
    if ("linear" in sys.argv):
        centroids = [np.random.random(IMAGE_SIZE) for _ in range(k)]
    elif ("gaussian" in sys.argv):
        centroids = [np.random.normal(0.0, 1.0, IMAGE_SIZE) for _ in range(k)]
    else:
        centroids = [np.random.randint(0, 255, IMAGE_SIZE) for _ in range(k)]

    n = x.shape[0]
    first_run = True
    iter_count = 0
    membership = np.zeros((k, n), dtype=int)
    cluster_change_counts = [n+1 for i in range(k)]
    sse_values = []

    while (any(count > stop_threshold for count in cluster_change_counts)):
        # Keep track of how many points have changed their clusters
        cluster_change_counts = [0 for i in range(k)]

        # Form the clusters
        for i in range(n):
            max_similarity = 0.0
            cluster_index = None

            # Find the best closest cluster
            for j in range(k):
                similarity = similarity_function(x[i], centroids[j])
                if (similarity > max_similarity):
                    max_similarity = similarity
                    cluster_index = j
            
            # Update the membership matrix and change counts
            for j in range(k):
                new_membership = 1 if (cluster_index == j) else 0
                if (membership[j][i] != new_membership):
                    cluster_change_counts[j] += 1
                membership[j][i] = new_membership

        # Recompute the centroids
        for j in range(k):
            member_number = sum([membership[j][i] for i in range(n)])
            if (member_number != 0):
                centroids[j] = sum([membership[j][i] * x[i] for i in range(n)]) / member_number

        # Be sure that each cluster has at least one change in the first iteration.
        # Otherwise, one cluster may have zero point. 
        if (first_run and 0 in cluster_change_counts):
            print("One cluster is subject to zero change in the first run. Probably it has no point. Restarting the algorithm...")
            centroids = [np.random.random(IMAGE_SIZE) for _ in range(k)]
            membership = np.zeros((k, n), dtype=int)
            iter_count = 0
            continue
        else:
            first_run = False
            iter_count += 1 

        # Calculate SSE and print the current state
        sse = sum_squared_error(train_x, centroids, membership)
        sse_values.append(sse)
        print("%d. Cluster Change Counts: %s\t\tSSE: %.2f" % (iter_count, cluster_change_counts, sse))


    print("Algorithm has stopped")
    return centroids, sse_values, membership

# Loads the MNIST data set given in binary format at the target path.
# Returns a np.ndarray(shape=(image_number, IMAGE_SIZE+1), dtype=np.float32).
def load_data(binary_file_path: str) -> np.ndarray:
    with open(binary_file_path, "rb") as binary_file:
        loaded_bytes = binary_file.read()
        byte_number = len(loaded_bytes)
        image_number = byte_number//(IMAGE_SIZE+1)

        if (image_number * (IMAGE_SIZE+1) != byte_number):
            raise Exception("Dimensions do not match")
        
        data = np.ndarray(shape=(image_number, IMAGE_SIZE+1), dtype=np.float32)
        for i in range(image_number):
            for j in range(IMAGE_SIZE+1):
                data[i][j] = loaded_bytes[i*(IMAGE_SIZE+1) + j]

    return data

# Load the data
data_path = "../data/mnist/"
print("Loading train data...")
train_data = load_data(data_path + "mnist_train.ubyte")
print("Loading test data...")
test_data = load_data(data_path + "mnist_test.ubyte") 

# The classes on which we work
classes = [2, 3, 8, 9]
class_number = len(classes)

# Process the train data
print("Processing train data...")
train_data = train_data[np.isin(train_data[:,0], classes)]
train_size = train_data.shape[0]
np.random.shuffle(train_data)
train_x = train_data[:, 1:]
train_y = train_data[:, 0]

# Process the test data
print("Processing test data...")
test_data = test_data[np.isin(test_data[:,0], classes)]
test_size = test_data.shape[0]
np.random.shuffle(test_data)
test_x = test_data[:, 1:]
test_y = test_data[:, 0]

# Linear normalization to [0.0, 1.0]
if ("linear" in sys.argv):
    print("Normalizing Linearly...")
    for x in [train_x, test_x]:
        image_number = x.shape[0]
        x /= 255.0
# Gaussian normalization to mean = 0, std = 1
elif ("gaussian" in sys.argv):
    print("Normalizing with Gaussian Distribution...")
    for x in [train_x, test_x]:
        image_number = x.shape[0]
        for i in range(image_number):
            mean = np.mean(x[i])
            std = np.std(x[i])
            x[i] = (x[i] - mean) / std
# No normalization
else:
    print("No Normalization...")

# Choose the similarity function
if ("cos" in sys.argv):
    similarity_function = cosine_similarity
    print("Using Cosine Similarity...")
elif ("euclid" in sys.argv):
    similarity_function = euclidean_similarity
    print("Using Euclidean Similarity...")
else: # default
    similarity_function = euclidean_similarity
    print("Using Default Similarity Measure that is Euclidean Similarity...")

# K-means returns k centroids, but we don't know which centroid corresponds to which class.
# This function returns a list containing classes respectively.
# For instance if [C8, C2, C3, C9] is list of centroids that we got from the K-means algorithm,
# this function returns [8, 2, 3, 9].
def find_cluster_class_pairs() -> tuple[list, np.ndarray]:
    matrix = np.zeros((class_number, class_number), dtype=float)
    for i in range(train_size):
        cls = classes.index(train_y[i])
        for cluster in range(class_number):
            similarity = similarity_function(centroids[cluster], train_x[i])
            matrix[cls][cluster] += similarity

    cluster_to_class = []
    for cluster in range(class_number):
        max_similarity = 0.0
        optimum_class = None
        for cls in range(class_number):
            if (classes[cls] not in cluster_to_class and matrix[cls][cluster] > max_similarity):
                max_similarity = matrix[cls][cluster]
                optimum_class = cls
        cluster_to_class.append(classes[optimum_class])

    return cluster_to_class, matrix

# Predicts a single point by using the centroids.
# Returns the predicted class.
def predict(point: np.ndarray, cluster_to_class: list) -> int:
    max_similarity = 0.0
    optimum_cluster = None
    for cluster in range(class_number):
        similarity = similarity_function(point, centroids[cluster])
        if (similarity > max_similarity):
            max_similarity = similarity
            optimum_cluster = cluster
    return cluster_to_class[optimum_cluster]

# Runs the algorithm on the test data.
# Returns the percent of the accuracy.
def test_model(cluster_to_class: list) -> float:
    correct_prediction_count = 0
    for i in range(test_size):
        predicted = predict(test_x[i], cluster_to_class)
        if (predicted == test_y[i]):
            correct_prediction_count += 1
    
    return 100.0 * correct_prediction_count / test_size

centroids, sse_values, embership = k_means_clustering(train_x, class_number, similarity_function)
cluster_to_class, matrix = find_cluster_class_pairs()
accuracy_percent = test_model(cluster_to_class)
plot_line_graph(sse_values, "SSE", "Iteration", "SSE")
plot_images(centroids, 2, class_number//2)

print()
print("Matrix of Sum of Similarities for Class(row)-Cluster(column):\n%s" % matrix)
print("Cluster to Class Map: %s" % cluster_to_class)
print("Accuracy on Test Set: %.2f %%" % accuracy_percent)

