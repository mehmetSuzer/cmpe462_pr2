import numpy as np
from cvxopt import matrix, solvers



class LinearSVM:
    def __init__(self, C=0.1, tol=1e-3, max_iter=None):
        self.C = C  # regularization parameter
        self.tol = tol  # tolerance for stopping criteria
        self.max_iter = max_iter  # maximum number of iterations
        self.weights = None
        self.biases = None
        self.classes = None
        self.num_classes = 0
        self.max_iter = max_iter


    def fit(self, X, y):
        self.classes = np.sort(np.unique(y))
        self.num_classes = len(self.classes)
        self.weights = np.zeros((self.num_classes, X.shape[1]))
        self.biases = np.zeros(self.num_classes)

        for i, cls in enumerate(self.classes):
            binary_y = np.where(y == cls, 1, -1)
            self.biases[i], self.weights[i] = self._train_one_vs_all(X, binary_y)

    def _train_one_vs_all(self, X, y):
        n_samples, n_features = X.shape


        P = np.zeros((n_features+1,n_features+1))

        for i in range(1,n_features+1):
            for j in range(1,n_features+1):
                if i == j:
                    P[i,j] = 1
        A = np.zeros((n_features+1,n_features+1))
        b = np.zeros(n_features+1)
        q = np.zeros(n_features+1)
        h = -np.ones(n_samples)
        G = np.zeros((n_samples,n_features+1))
        for i in range(n_samples):
            G[i,0] = y[i]
            G[i,1:] = X[i] * y[i]
        P = matrix(P)
        q = matrix(q)
        G = matrix(-G)
        h = matrix(h)
        if self.max_iter:
            options = {'maxiters':self.max_iter}
            solution = solvers.qp(P, q, G, h,options=options)
        else:
            solution = solvers.qp(P,q,G,h)
        return solution['x'][0], np.array(solution['x'][1:])[:,0]

    def predict(self,X):
        scores = np.zeros((self.num_classes,X.shape[0]))
        for i in range(self.num_classes):
            scores[i] = np.dot(X,self.weights[i]) + self.biases[i]
        result = np.argmax(scores,axis=0)
        for i in range(len(result)):
            result[i] = self.classes[result[i]]
        return result



class NonLinearSVM:

    def __init__(self, C=None,sigma=None):
        self.C = C
        self.sigma = sigma
        self.biases = None
        if self.C is not None: self.C = float(self.C)
        if self.sigma is None: self.sigma = 5.0

    @staticmethod
    def gaussian_kernel(x, y, sigma):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    def fit(self, X, y):
        self.classes = np.sort(np.unique(y))
        self.num_classes = len(self.classes)
        self.biases = np.zeros(self.num_classes)
        self.svs = []
        self.alphas = []
        self.sv_ys = []

        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.gaussian_kernel(X[i], X[j],self.sigma)

        for i, cls in enumerate(self.classes):
            binary_y = np.where(y == cls, 1, -1)
            sv, alpha,sv_y, self.biases[i] = self.train_one_vs_all(X, binary_y, K)
            self.svs.append(sv)
            self.alphas.append(alpha)
            self.sv_ys.append(sv_y)

    def train_one_vs_all(self, X, y, K):
        n_samples, n_features = X.shape
        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_samples) * -1)
        A = y.astype(np.double)
        A = matrix(A,(1,n_samples))
        b = matrix(np.zeros(1))
        G = matrix(-np.identity(n_samples))
        h = matrix(np.zeros(n_samples))

        # if self.C is None:
        #     G = matrix(np.diag(np.ones(n_samples) * -1))
        #     h = matrix(np.zeros(n_samples))
        # else:
        #     tmp1 = np.diag(np.ones(n_samples) * -1)
        #     tmp2 = np.identity(n_samples)
        #     G = matrix(np.vstack((tmp1, tmp2)))
        #     tmp1 = np.zeros(n_samples)
        #     tmp2 = np.ones(n_samples) * self.C
        #     h = matrix(np.hstack((tmp1, tmp2)))

        solution = solvers.qp(P,q,G,h,A,b)
        a = np.ravel(solution['x'])
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        bias = 0
        for n in range(len(a[sv])):
            bias += y[sv][n]
            bias -= np.sum(a[sv] * y[sv] * K[ind[n], sv])
        bias /= len(a[sv])

        return X[sv], a[sv], y[sv], bias


    def predict(self, X):
        pred = np.zeros((self.num_classes, len(X)))
        for j in range(self.num_classes):
            y_predict = np.zeros(len(X))
            alpha = self.alphas[j]
            sv_ys = self.sv_ys[j]
            svs = self.svs[j]
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(alpha, sv_ys, svs):
                    s += a * sv_y * self.gaussian_kernel(X[i], sv,self.sigma)
                y_predict[i] = s
            pred[j] = y_predict + self.biases[j]
        result = np.argmax(pred, axis=0)
        for i in range(len(result)):
            result[i] = self.classes[result[i]]
        return result
    def calculateWeights(self):
        weights = np.zeros((self.num_classes,self.svs[0].shape[-1]))
        for i in range(self.num_classes):
            alpha = self.alphas[i]
            sv_ys = self.sv_ys[i]
            svs = self.svs[i]
            weight = np.zeros(svs.shape[-1])
            for a, sv_y, sv in zip(alpha, sv_ys, svs):
                weight += a * sv_y * sv
            weights[i] = weight
        return weights





