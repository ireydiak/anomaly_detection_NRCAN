from matplotlib import pyplot as plt
import numpy as np


def plot_losses(train_loss, val_loss, save_path='./'):
    """

    Parameters
    ----------
    train_loss
    val_loss
    save_path: path where to save figure

    Returns
    -------

    """
    num_data = range(len(train_loss))

    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)
    # ax2 = f.add_subplot(122)

    # loss plot
    ax1.plot(num_data, train_loss, label='Training loss')
    ax1.plot(num_data, val_loss, label='Test loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('# epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    f.savefig(save_path + 'fig.png')
    plt.show()


def plot_2D_latent(X, y, save_path='./'):
    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)

    for label in np.unique(y):
        mask = y == label
        ax1.scatter(X[mask, 0], X[mask, 1], label=str(label))

    ax1.set_title('Representation in latent space')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    f.savefig(save_path + 'fig_latent.png')
    plt.show()


def plot_energy_percentile(energies, save_path='./'):
    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)

    q = np.linspace(0, 100, 100)
    ax1.plot(np.percentile(energies, q), q, label='Training loss')
    ax1.set_title('Percentage vs Energy')
    ax1.set_xlabel('percentage')
    ax1.set_ylabel('energy')
    ax1.legend()
    f.savefig(save_path + 'fig_energy_percent.png')
    plt.show()


def plot_1D_latent_vs_loss(X, y, losses_items, save_path='./'):
    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)

    for label in np.unique(y):
        mask = y == label
        ax1.scatter(X[mask, 0], losses_items[mask], label=str(label))

    ax1.set_title('Representation in latent space')
    ax1.set_xlabel('z')
    ax1.set_ylabel('loss')
    ax1.legend()
    f.savefig(save_path + 'fig_latent.png')
    plt.show()


def plot_3D_latent(z, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z[:, 1], z[:, 0], z[:, 2], c=labels.astype(int))
    ax.set_xlabel('Encoded')
    ax.set_ylabel('Euclidean')
    ax.set_zlabel('Cosine')
    plt.show()
