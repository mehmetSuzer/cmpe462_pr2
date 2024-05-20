import cv2
import numpy as np

def extract_sift(x_train,x_test,num_dim):
    train_image_descriptors = []
    test_image_descriptors = []
    sift = cv2.SIFT_create()
    cv2.resize

    all_desc = []

    train_none_list = []
    test_none_list = []

    for i in range(len(x_train)):
        keypoints, descriptors = sift.detectAndCompute(x_train[i], None)
        train_image_descriptors.append(np.array(descriptors))
        if descriptors is None:
            train_none_list.append(i)
            continue
        for desc in descriptors:
            all_desc.append(desc)

    for i in range(len(x_test)):
        keypoints, descriptors = sift.detectAndCompute(x_test[i], None)
        test_image_descriptors.append(np.array(descriptors))
        if descriptors is None:
            test_none_list.append(i)
            continue
        for desc in descriptors:
            all_desc.append(desc)

    test_none = np.array(test_none_list)
    train_none = np.array(train_none_list)
    train_desc = train_image_descriptors
    test_desc = test_image_descriptors

    from sklearn.cluster import KMeans

    all_desc = np.array(all_desc)

    kmeans = KMeans(n_clusters=int(num_dim))
    kmeans.fit(all_desc)

    centers = kmeans.cluster_centers_

    train_histograms = []

    for i in range(len(train_desc)):
        hist = np.zeros(int(num_dim))
        if i in train_none:
            train_histograms.append(hist)
            continue
        for desc in train_desc[i]:
            index = np.argmin(np.mean((centers - desc) ** 2, axis=1))
            hist[index] += 1
        hist = hist / np.sum(hist)
        train_histograms.append(hist)

    test_histograms = []

    for i in range(len(test_desc)):
        hist = np.zeros(int(num_dim))
        if i in test_none:
            test_histograms.append(hist)
            continue
        for desc in test_desc[i]:
            index = np.argmin(np.mean((centers - desc) ** 2, axis=1))
            hist[index] += 1
        hist = hist / np.sum(hist)
        test_histograms.append(hist)

    train_histograms = np.array(train_histograms)
    test_histograms = np.array(test_histograms)
    return train_histograms, test_histograms

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_dim', help='number of dimensions the features will have', default=32, const=32,
                        nargs='?')
    parser.add_argument('--data_location',
                        help='please specify where data is unless you will use tensorflow, note that the script assumes there are files named x_train.npy, y_train.npy, x_test.npy and y_test.npy')
    parser.add_argument('--use_tf', help='if used, the data is downloaded using tensorflow', action='store_true')
    parser.add_argument('--extract_location',
                        help="please specify where the extracted features will be stored, the script will dump the features into a file named train_histograms.npy and test_histograms.npy",
                        required=True)
    args = parser.parse_args()

    if args.use_tf:
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        x_train = np.load(args.data_location + '/x_train.npy')
        y_train = np.load(args.data_location + '/y_train.npy')
        x_test = np.load(args.data_location + '/x_test.npy')
        y_test = np.load(args.data_location + '/y_test.npy')
    #
    x_train = x_train[(y_train == 2) | (y_train == 3) | (y_train == 8) | (y_train == 9)]
    y_train = y_train[(y_train == 2) | (y_train == 3) | (y_train == 8) | (y_train == 9)]
    x_test = x_test[(y_test == 2) | (y_test == 3) | (y_test == 8) | (y_test == 9)]
    y_test = y_test[(y_test == 2) | (y_test == 3) | (y_test == 8) | (y_test == 9)]

    train_histograms, test_histograms = extract_sift(x_train,x_test,args.num_dim)

    np.save(args.extract_location + '/train_histograms.npy', train_histograms)
    np.save(args.extract_location + '/test_histograms.npy', test_histograms)
