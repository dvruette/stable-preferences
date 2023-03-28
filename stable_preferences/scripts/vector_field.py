import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(vectors):
    y, x = np.meshgrid(np.arange(vectors.shape[1]), np.arange(vectors.shape[0]))
    plt.quiver(x, y, vectors[..., 0], vectors[..., 1])

def main():
    pos_points = np.random.multivariate_normal((3, 5), [[5, 0], [0, 2]], size=(30,))
    neg_points = np.random.multivariate_normal((-5, -5), [[2, -2.25], [-2.25, 2]], size=(20,))
    neg_points = np.concatenate([neg_points, np.random.uniform(-8, 6, size=(40, 2))], axis=0)
    xs, ys = np.linspace(-10, 10, 30), np.linspace(-10, 10, 30)
    vectors = np.zeros((len(xs), len(ys), 2))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            pos_dists = np.linalg.norm(pos_points - p, axis=1)
            neg_dists = np.linalg.norm(neg_points - p, axis=1)
            pos_dists = 1 / (pos_dists**2 + 1)
            # neg_dists = 1 / (neg_dists**2 + 1)
            # pos_dists = np.exp(-pos_dists**2 / 4)
            neg_dists = np.exp(-neg_dists**2 / 10)
            pos_dists /= np.sum(pos_dists)
            neg_dists /= np.sum(neg_dists)

            pos_avg = np.sum(pos_points * pos_dists[:, None], axis=0)
            neg_avg = np.sum(neg_points * neg_dists[:, None], axis=0)
            d = pos_avg - neg_avg
            vectors[i, j] = d / np.linalg.norm(d)

    y_grid, x_grid = np.meshgrid(xs, ys)
    plt.quiver(x_grid, y_grid, vectors[..., 0], vectors[..., 1], color="blue", alpha=0.5)
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color="green", s=4)
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color="red", s=4)
    plt.show()

if __name__ == "__main__":
    main()
