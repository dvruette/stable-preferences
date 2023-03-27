from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def field_strength(distance, kernel: Literal["inverse", "rbf", "grad_inverse"] = "grad_inverse"):
    """
    Compute the field strength at a given distance.
    """
    if kernel == "inverse":
        return 1 / (distance**2 + 1)
    elif kernel == "grad_inverse":
        # compute gradient magnitude of 1 / (distance**2 + 1)
        # this is equivalent to 1 / (distance**2 + 1) - 2 / (distance**4 + 1)
        return -2*distance / ((distance**2 + 1)**2)
    else:
        return np.exp(-(distance ** 2)/10)

def field_gradient(x, p):
    """
    Compute the gradient of the field strength at a given point.
    """
    # compute the gradient of the field strength
    # this is equivalent to the gradient of 1 / (distance**2 + 1)
    # the gradient of 1 / (distance**2 + 1) is -2 * (x - p) / (distance**4 + 1)
    # the gradient of the field strength is -2 * (x - p) / (distance**4 + 1) if the point is closer to an attractor
    # the gradient of the field strength is 2 * (x - p) / (distance**4 + 1) if the point is closer to a repulsor
    # the gradient of the field strength is 0 if the point is closer to both an attractor and a repulsor
    return -2 * (x - p) / ((np.linalg.norm(x - p)**2 + 1)**2)

def main():

    """
    Visualize a field of attractors and repulsors.
    """

    # define the field
    # grid is 100 x 100
    # attractors are sampled with mean (-1, -1) and std 0.1
    # repulsors are sampled with mean (1, 1) and std 0.1
    num_points = 20
    width, height = 100, 100
    std = 0.25
    xs = np.linspace(-2, 2, width)
    ys = np.linspace(-2, 2, height)
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    attractors = np.random.normal(-1, std, size=(num_points, 2))
    repulsors = np.random.normal(1, std, size=(num_points, 2))
    # also add some uniform repulsors
    repulsors = np.concatenate([repulsors, np.random.uniform(-2, 2, size=(num_points, 2))], axis=0)

    # compute the field
    # field strength is 1 / distance
    # field strength is negative if the point is closer to an attractor
    # field strength is positive if the point is closer to a repulsor
    field = np.zeros((width, height))
    # for attractor in attractors:
    #     field += field_strength(np.linalg.norm(grid - attractor, axis=1)).reshape(width, height)
    # for repulsor in repulsors:
    #     field -= field_strength(np.linalg.norm(grid - repulsor, axis=1)).reshape(width, height)
    gradient = np.zeros((width, height, 2))
    for attractor in attractors:
        gradient += np.array([field_gradient(x, attractor) for x in grid]).reshape(width, height, 2)
    for repulsor in repulsors:
        gradient -= np.array([field_gradient(x, repulsor) for x in grid]).reshape(width, height, 2)
    field = np.linalg.norm(gradient, axis=2)

    # Visualize the field
    # attractors are shown in green
    # repulsors are shown in red
    # field is shown as a heatmap
    plt.scatter(attractors[:, 0], attractors[:, 1], color="green", s=1)
    plt.scatter(repulsors[:, 0], repulsors[:, 1], color="red", s=1)
    plt.imshow(field.T, extent=(-2, 2, -2, 2), origin="lower")
    # show color bar
    plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    main()
