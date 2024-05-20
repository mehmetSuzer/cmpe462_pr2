from models import LinearSVM, NonLinearSVM
from feature_extract import extract_sift
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_location',help='please specify where data is unless you will use tensorflow, note that the script assumes there are files named x_train.npy, y_train.npy, x_test.npy and y_test.npy')
parser.add_argument('--use_tf',help='if used, the data is downloaded using tensorflow',action='store_true')
parser.add_argument('--use_sift',help='if used, the sift features are used',action='store_true')
parser.add_argument('--num_dim', help='number of dimensions the sift features will have', default=32, const=32,
                    nargs='?')
parser.add_argument('--C', help='Regularization parameter', default=1.0, const=1.0,nargs='?')
parser.add_argument('--gamma', help='Kernel coefficient for rbf')
parser.add_argument('--use_sklearn',help='if used, scikit-learn SVMs will be used',action='store_true')
parser.add_argument('--nonlinear',help='if used, non-linear svm will be used',action='store_true')
parser.add_argument('--grid_search',help='if used a grid search is performed to find the best hyperparameters',action='store_true')
parser.add_argument('--plot',help='if used, plots the support vectors',action='store_true')
args = parser.parse_args()

if args.use_tf:
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    x_train = np.load( args.data_location + '/x_train.npy')
    y_train = np.load(args.data_location + '/y_train.npy')
    x_test = np.load(args.data_location + '/x_test.npy')
    y_test = np.load(args.data_location + '/y_test.npy')


x_train = x_train[(y_train == 2) | (y_train == 3) | (y_train == 8 )| (y_train == 9)]
y_train = y_train[(y_train == 2) | (y_train == 3) | (y_train == 8 )| (y_train == 9)]
x_test = x_test[(y_test == 2) | (y_test == 3) | (y_test == 8 )| (y_test == 9)]
y_test = y_test[(y_test == 2) | (y_test == 3) | (y_test == 8 )| (y_test == 9)]


if args.use_sift:
    x_train, x_test = extract_sift(x_train,x_test,args.num_dim)
else:
    x_train = np.array(x_train,dtype=np.float32)
    x_test = np.array(x_test,dtype=np.float32)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    x_train /= 255.0
    x_test /= 255.0


if args.use_sklearn:
    from sklearn.svm import SVC
    if args.grid_search:
        from sklearn.model_selection import GridSearchCV
        if args.nonlinear:
            param_grid = {'C': [0.1, 1, 10, 100],
                          'gamma': [0.001, 0.01, 0.1, 1]}
            svm = SVC(kernel='rbf')
            grid_search = GridSearchCV(svm, param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            best_C = grid_search.best_params_['C']
            best_gamma = grid_search.best_params_['gamma']
            svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
        else:
            param_grid = {'C': [0.1, 1, 10, 100]}
            svm = SVC(kernel='linear')
            grid_search = GridSearchCV(svm, param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            best_C = grid_search.best_params_['C']
            svm = SVC(kernel='linear', C=best_C)
    else:
        if args.nonlinear:
            if args.gamma == None:
                svm = SVC(kernel='rbf', C=args.C)
            else:
                svm = SVC(kernel='rbf', C=args.C, gamma=args.gamma)
        else:
            svm = SVC(kernel='linear', C=args.C)

else:
    if args.nonlinear:
        svm = NonLinearSVM()
    else:
        svm = LinearSVM()
if args.plot:
    import matplotlib.pyplot as plt
    vectors = svm.calculateWeights()
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(vectors[0].reshape((28, 28)))
    axes[0, 0].set_title('SVECTOR 2')
    axes[0, 1].imshow(vectors[1].reshape((28, 28)))
    axes[0, 1].set_title('SVECTOR 3')
    axes[1, 0].imshow(vectors[2].reshape((28, 28)))
    axes[1, 0].set_title('SVECTOR 8')
    axes[1, 1].imshow(vectors[3].reshape((28, 28)))
    axes[1, 1].set_title('SVECTOR 9')
    plt.show()


svm.fit(x_train,y_train)
train_predictions = svm.predict(x_train)
test_predictions = svm.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Accuracy for : Train = {train_accuracy:.4f}, Test = {test_accuracy:.4f}")
