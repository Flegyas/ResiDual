import matplotlib.pyplot as plt
import torch


def blocked_heatmap(
    data,
    y_labels,
    block_size,
    height: int = 30,
    cmap: str = "Blues",
    title: str = "Blocked Heatmap",
):
    n, d = data.shape
    block_width = 4
    block_height = block_size // block_width  # int(math.sqrt(block_size))

    data_blocked = torch.as_tensor(data).reshape(
        n, d // block_size, block_height, block_width
    )
    data_blocked = (
        data_blocked.permute(0, 2, 1, 3).flatten(end_dim=1).flatten(start_dim=1)
    )
    relative_width = data_blocked.shape[0] / data_blocked.shape[1]
    width = height * relative_width
    fig, ax = plt.subplots(figsize=(height, width))
    im = ax.imshow(data_blocked, cmap=cmap)
    block_num = d // block_size
    y_tick_positions = [(i * block_height + block_height / 2) - 0.5 for i in range(n)]
    ax.set_yticks(y_tick_positions, labels=y_labels, fontsize=26)
    x_tick_positions = [
        (i * block_width + block_width / 2) - 0.5 for i in range(block_num)
    ]
    ax.set_xticks(x_tick_positions, labels=range(block_num), fontsize=26)

    # add colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
    cb.ax.tick_params(labelsize=30)

    for i in range(1, n):
        ax.axhline(i * block_height - 0.5, color="black", linewidth=2)
    for i in range(1, block_num):
        ax.axvline(i * block_width - 0.5, color="black", linewidth=2)

    # plt.tight_layout()
    return fig


if __name__ == "__main__":
    datasets = [
        "ImageNet",
        "ImageNet Sketch",
        "MNIST",
        "SVHN",
        "ColoredMNIST",
        "Sun397",
        "EuroSAT",
        "GTSRB",
        "CIFAR10",
        "StanfordCars",
        "DTD",
        "RESISC45",
        "Random",
    ]
    d = 384  # Number of units in the residual stream
    block_size = 16  # Group units into blocks of block_size x block_size (layers)
    n = len(datasets)

    # Dummy data
    data = torch.randn(n * d).reshape(n, d) + 1
    plt.title("Original data")
    plt.imshow(data, aspect="auto", cmap="Blues", interpolation="nearest")
    plt.yticks(range(n), datasets)
    plt.show()

    # Generate the heatmap
    blocked_heatmap = blocked_heatmap(
        data=data, block_size=block_size, y_labels=datasets
    )
    blocked_heatmap.show()
