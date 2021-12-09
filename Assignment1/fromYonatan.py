import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.ticker as mt
#from GradNet import gradient_test_loss_theta, create_theta
from Neural_Network import NeuralNetwork

"""
# Utils and shuffle the input:
dict1 = loadmat('SwissRollData.mat')
Xt_1 = dict1["Yt"]
Xv_1 = dict1["Yv"]
Ct_1 = dict1["Ct"]
Cv_1 = dict1["Cv"]
dict2 = loadmat('PeaksData.mat')
Xt_2 = dict2["Yt"]
Xv_2 = dict2["Yv"]
Ct_2 = dict2["Ct"]
Cv_2 = dict2["Cv"]
dict3 = loadmat('GMMData.mat')
Xt_3 = dict3["Yt"]
Xv_3 = dict3["Yv"]
Ct_3 = dict3["Ct"]
Cv_3 = dict3["Cv"]
"""

def shuffle(X, C):
    A = np.vstack([X, C])
    np.random.shuffle(A.transpose())
    X = A[0:X.shape[0], :]
    C = A[X.shape[0]:, :]
    return X, C


def add_row_of_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])


"""Xt_1, Ct_1 = shuffle(Xt_1, Ct_1)
Xt_2, Ct_2 = shuffle(Xt_2, Ct_2)
Xt_3, Ct_3 = shuffle(Xt_3, Ct_3)

Xt_1 = add_row_of_ones(Xt_1)
Xv_1 = add_row_of_ones(Xv_1)
Xt_2 = add_row_of_ones(Xt_2)
Xv_2 = add_row_of_ones(Xv_2)
Xt_3 = add_row_of_ones(Xt_3)
Xv_3 = add_row_of_ones(Xv_3)"""


def soft_max_regression(X, C, w):
    m = X.shape[1]
    num_of_labels = C.shape[0]
    X_t = X.transpose()
    tmp_sum = np.zeros(m)

    for j in range(num_of_labels):
        tmp_sum += np.exp(X_t @ w[:, j])
    sum = 0
    for k in range(num_of_labels):
        sum += C[k, :] @ np.log(np.exp(X_t @ w[:, k]) / tmp_sum)
    return (-1 / m) * sum


def soft_max_regression_gradient_w_p(X, C, w, p):
    m = X.shape[1]
    num_of_labels = w.shape[1]
    X_transpose = X.transpose()
    tmp_vector = np.zeros(m)

    for j in range(num_of_labels):
        tmp_vector += np.exp(X_transpose @ w[:, j])
    matrix = np.linalg.inv(np.diag(tmp_vector))
    vector = matrix @ np.exp(X_transpose @ w[:, p]) - C[p, :].transpose()
    return (1/m) * X @ vector


def sgd(Xt, Xv, Ct, Cv, w, eta, batch_size, plot='no'):
    m = Xt.shape[1]
    n = Xt.shape[0]
    num_of_labels = w.shape[1]
    num_of_epochs = 25
    iterations = int(m / batch_size)
    sum = np.zeros(w.shape)
    x_axis_t = np.zeros(num_of_epochs)
    y_axis_t = np.zeros(num_of_epochs)
    x_axis_v = np.zeros(num_of_epochs)
    y_axis_v = np.zeros(num_of_epochs)

    for epoch in range(num_of_epochs):
        for t in range(1, iterations+1):
            X_batch = Xt[:, (t - 1) * batch_size:t * batch_size]
            C_batch = Ct[:, (t - 1) * batch_size:t * batch_size]
            grad_wi = np.zeros((n, num_of_labels))
            for p in range(num_of_labels):
                grad_wi[:, p] = soft_max_regression_gradient_w_p(X_batch, C_batch, w, p)
            w -= eta * grad_wi
            sum = sum + w
        x_axis_t[epoch] = epoch
        y_axis_t[epoch] = success_percentage_of_w(Xt, Ct, sum / ((epoch+1)*iterations))
        x_axis_v[epoch] = epoch
        y_axis_v[epoch] = success_percentage_of_w(Xv, Cv, sum / ((epoch+1)*iterations))
    if plot == 'yes':
        plot_graph_accuracy_vs_epoch(x_axis_t, y_axis_t, x_axis_v, y_axis_v,
                                     batch_size, eta, num_of_epochs)
    return sum / (num_of_epochs*iterations)


def gradient_test(X, C, w):
    iterations = 20
    n = X.shape[0]
    num_of_labels = C.shape[0]
    d = np.random.rand(w.shape[0], w.shape[1])
    d_flat = d.transpose().reshape(-1)
    a_all = np.zeros(iterations-9)
    epsilon_all = np.zeros(iterations-9)
    b_all = np.zeros(iterations-9)
    i_all = np.zeros(iterations-9)

    for i in range(10, iterations+1):
        epsilon_i = 0.5 ** i
        f_x = soft_max_regression(X, C, w)
        f_x_ed = soft_max_regression(X, C, w + epsilon_i * d)
        grad_x = np.ones((n, num_of_labels))
        for p in range(num_of_labels):
            grad_x[:, p] = soft_max_regression_gradient_w_p(X, C, w, p)
        grad_x_flat = grad_x.transpose().reshape(-1)
        a = abs(f_x_ed - f_x)
        b = abs(f_x_ed - f_x - epsilon_i * d_flat @ grad_x_flat)
        a_all[i-10] = a
        epsilon_all[i-10] = epsilon_i
        b_all[i-10] = b
        i_all[i-10] = i
    plot_graph_gradient_test(a_all, b_all, i_all, epsilon_all, iterations)


def plot_graph_gradient_test(a_all, b_all, i_all, epsilon_all, iterations):
    quadratic_epsilon = epsilon_all ** 2
    plt.plot(i_all, a_all, color='green', label=r'|f(x+$\epsilon$d)-f(x)|', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='green', markersize=0)
    plt.plot(i_all, b_all, color='blue', label=r'|f(x+$\epsilon$d)-f(x)-$\epsilon$$d^Tgrad(x)$|', linestyle='solid',
             linewidth=1, marker='o', markerfacecolor='blue', markersize=0)

    plt.plot(i_all, epsilon_all, color='red', label=r'$\epsilon$', linestyle='dashed', linewidth=1,
             marker='o', markerfacecolor='red', markersize=0)

    plt.plot(i_all, quadratic_epsilon, color='orange', label=r'$\epsilon^2$', linestyle='dashed', marker='o',
             markerfacecolor='orange', markersize=0)
    plt.xlim(10, iterations)
    plt.yscale('log', base=10)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.title(" Gradient Test  ")
    plt.show()


def plot_graph_accuracy_vs_epoch(x_axis_t, y_axis_t, x_axis_v, y_axis_v, batch_size, eta, num_of_epochs):
    plt.plot(x_axis_t, y_axis_t, color='green', label='Training Set', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='green', markersize=0)
    plt.plot(x_axis_v, y_axis_v, color='blue', label='Validation Set',  linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=0)
    # for limiting the x axis from 1
    x_axis_1 = np.zeros(2)
    y_axis_1 = np.zeros(2)
    y_axis_1[1] = 1
    plt.plot(x_axis_1, y_axis_1, color='blue', linestyle='solid', linewidth=0,
             marker='o', markerfacecolor='blue', markersize=0)
    plt.xlim(1, num_of_epochs)
    m = mt.PercentFormatter(1)
    plt.gca().yaxis.set_major_formatter(m)
    plt.xlabel("Number of Epochs")
    plt.ylabel('Accuracy Percentage')
    plt.legend()
    plt.title('Learning rate= %.2f' % eta + '       Accuracy VS Epochs        Batch size =%d' % batch_size)
    plt.show()


def plot_per_parameter(Xt, Xv, Ct, Cv):
    learning_rates = np.array([0.01, 0.1])
    batch_sizes = np.array([25, 50, 100, 200])
    cart_prod = [(a, b) for a in learning_rates for b in batch_sizes]
    for pair in cart_prod:
        w = np.zeros((Xt.shape[0], Ct.shape[0]))
        sgd(Xt, Xv, Ct, Cv, w, pair[0], pair[1], 'yes')


def C_to_Y(C):
    m = C.shape[1]
    num_of_labels = C.shape[0]
    Y = np.zeros(m)

    for p in range(num_of_labels):
        for i in range(m):
            if C[p, i] == 1:
                Y[i] = p
    return Y.transpose()


def success_percentage_of_w(X, C, w):
    Y = C_to_Y(C)
    m = X.shape[1]
    num_of_labels = C.shape[0]
    num_of_successes = 0

    for i in range(m):
        x_i = X[:, i]
        our_predicted_y_i = 0
        tmp_max = float('-inf')
        for j in range(num_of_labels):
            current = x_i.transpose() @ w[:, j]
            if current > tmp_max:
                our_predicted_y_i = j
                tmp_max = current
        if our_predicted_y_i == Y[i]:
            num_of_successes += 1
    return num_of_successes / m



# Part 1: Task 1 - Gradient Test:
Xtrain = np.array([[5, 5.2, 1], [1, 3, 1], [10, 13.01, 1], [1.1, 0.09, 1], [8, 1, 1], [2, 1.5, 1]]).transpose()
print(Xtrain)
Ctrain = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]).transpose()
print(Ctrain)
# Yt = np.array([[0], [0], [0], [1], [1], [1]])
my_w = np.random.rand(3, 2)
gradient_test(Xtrain, Ctrain, my_w)

"""
# Part 1: Task 2 - Accuracy vs Epoch:
plot_per_parameter(Xt_1, Xv_1, Ct_1, Cv_1)
plot_per_parameter(Xt_2, Xv_2, Ct_2, Cv_2)
plot_per_parameter(Xt_3, Xv_3, Ct_3, Cv_3)
"""


"""# Part 2: Task 1 - Jacobian tests
net = NeuralNetwork(5, 10, "tanh")
theta = net.create_theta(3, 2)
x = np.random.rand(3)
net.jacobian_test_x(x, theta)
net.jacobian_test_w(x, theta)
net.jacobian_test_b(x, theta)
net.transpose_test_x(x, theta)
net.transpose_test_w(x, theta)
net.transpose_test_b(x, theta)"""


"""# Part 2: Task 2 -
Xtrain = np.array([[0.5, 15, 1], [0.5, 3, 1], [0, 13.01, 1], [1, 0.01, 1], [1, 0.1, 1], [2, 0.5, 1]]).transpose()
Ctrain = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]).transpose()
# Yt = np.array([[0], [0], [0], [1], [1], [1]])
gradient_test_loss_theta(Xtrain, Ctrain, create_theta(3, 2, 3, 10))"""


"""
# Part 2: Task 3 -
net = NeuralNetwork(3, 100, "tanh")
net.plot_per_parameter(Xt_1, Xv_1, Ct_1, Cv_1)
net.plot_per_parameter(Xt_2, Xv_2, Ct_2, Cv_2)
net.plot_per_parameter(Xt_3, Xv_3, Ct_3, Cv_3)

net = NeuralNetwork(4, 100, "tanh")
net.plot_per_parameter(Xt_1, Xv_1, Ct_1, Cv_1)
#net.plot_per_parameter(Xt_2, Xv_2, Ct_2, Cv_2)
#net.plot_per_parameter(Xt_3, Xv_3, Ct_3, Cv_3)
"""