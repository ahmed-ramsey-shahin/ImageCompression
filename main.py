from matplotlib import image as img
import matplotlib.pyplot as plt
import numpy as np
import sys


def cost(x, centroids, idx):
    j = 0
    for i, _ in enumerate(x):
        j += np.sum(np.power((x[i, :] - centroids[idx[i].astype('int'), :]), 2))
    return (1 / x.shape[0]) * j


def preprocess_image(image_path):
    image = img.imread(image_path)
    original_image_shape = image.shape
    image = image.reshape((image.shape[0] * image.shape[1], image.shape[-1]))
    return image, original_image_shape


def init_centroids(x, k):
    indexes = np.random.randint(0, k - 1, (k,))
    return x[indexes]


def pick_closest_centroid(x, centroids):
    idx = np.zeros((x.shape[0], 1))
    for i, _ in enumerate(x):
        mu = np.zeros((centroids.shape[0], 1))
        for j, _ in enumerate(centroids):
            mu[j] = sum(np.power((x[i, :] - centroids[j, :]), 2))
        idx[i] = np.argmin(mu)
    return idx


def update_centroids(x, idx, k):
    centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        if idx[idx == i].shape[0] == 0:
            centroids[i, :] = x[np.random.randint(0, k - 1, (1,))[0], :]
            continue
        centroids[i, :] = (1 / idx[idx == i].shape[0]) * sum(x[np.where(idx == i)[0], :])
    return centroids


def run_kmeans(x, k, iterations):
    centroids = init_centroids(x, k)
    idx = pick_closest_centroid(x, centroids)
    for i in range(iterations):
        idx = pick_closest_centroid(x, centroids)
        centroids = update_centroids(x, idx, k)
    return centroids, idx


def kmeans(x, k, iterations):
    centroids = None
    idx = None
    j = 10e10
    for i in range(5):
        centroids_new, idx_new = run_kmeans(x, k, iterations)
        j_new = cost(x, centroids_new, idx_new)
        if j > j_new:
            centroids = centroids_new
            idx = idx_new
        print(i + 1)
    return centroids, idx


def compress(x, k):
    centroids, idx = kmeans(x, k, 20)
    idx = idx.reshape((idx.shape[0],))
    x = centroids[idx.astype('int')]
    return x


if __name__ == '__main__':
    X, original_shape = preprocess_image(sys.argv[1])
    K = int(sys.argv[2])
    compressed_image = compress(X, K)
    plt.imshow(compressed_image.reshape(original_shape))
    plt.show()
