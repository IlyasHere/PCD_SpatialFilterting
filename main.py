
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects

# 1. Manual 2D Convolution
def convolution2d(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    h, w = image.shape
    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Ukuran kernel harus ganjil.")

    padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)))
    kernel = np.flip(kernel, axis=(0, 1))
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# 2. Smoothing - Box Filter 3x3
def smoothing(image):
    kernel = np.ones((3, 3)) / 9
    return np.clip(convolution2d(image, kernel), 0, 255)

# 3. Sharpening - Laplacian 3x3
def sharpening(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    return np.clip(image + convolution2d(image, kernel), 0, 255)

# 4. Simple Terminal Output
def print_results(results):
    print("\n=== Python W6 - Spatial Filtering ===")
    for title, matrix in results:
        print(f"\n{title}:")
        for row in matrix:
            print("  " + " ".join(f"{v:>5.0f}" for v in row))
    h, w = results[0][1].shape
    print(f"\nUkuran Output : {h} x {w}")
    print("Status        : Berhasil\n")

# 5. Monochrome Palette
mono = LinearSegmentedColormap.from_list(
    "mono", ["#F0F0F0", "#C4C4C4", "#8E8E8E", "#575757", "#252525"]
)

# 6. Matrix Visualization
def draw_matrix(ax, data, title, vmin=None, vmax=None):
    data = np.asarray(data, dtype=float)
    vmin = data.min() if vmin is None else vmin
    vmax = data.max() if vmax is None else vmax
    if vmin == vmax:
        vmax = vmin + 1

    ax.imshow(data, cmap=mono, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    h, w = data.shape
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("X / Kolom", fontsize=10, labelpad=8)
    ax.set_ylabel("Y / Baris", fontsize=10, labelpad=8)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.tick_params(length=0, labelsize=9)

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(h):
        for j in range(w):
            value = data[i, j]
            norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
            r, g, b, _ = mono(norm)
            light = 0.2126*r + 0.7152*g + 0.0722*b
            color = "#111111" if light > 0.52 else "white"
            outline = "white" if light > 0.52 else "#202020"

            txt = ax.text(j, i, f"{value:.0f}", ha="center",
                          va="center", fontsize=10,
                          fontweight="bold", color=color)
            txt.set_path_effects([
                patheffects.withStroke(linewidth=0.8, foreground=outline)
            ])

    for spine in ax.spines.values():
        spine.set_color("#B3B3B3")

# 7. Balanced Horizontal Layout
def show_results(results):
    fig, axes = plt.subplots(
        1, 4, figsize=(19, 6), facecolor="#FAFAFA"
    )

    titles = [
        "Original Grayscale",
        "Manual 2D Convolution",
        "Smoothing - Box Filter",
        "Sharpening - Laplacian"
    ]

    for i, (ax, (_, data), title) in enumerate(zip(axes, results, titles)):
        if i == 1:
            draw_matrix(ax, data, title)
        else:
            draw_matrix(ax, data, title, vmin=0, vmax=255)

    fig.suptitle(
        "SPATIAL FILTERING",
        fontsize=22, fontweight="bold",
        y=0.94, color="#202020"
    )

    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [0.84, 0.84],
        transform=fig.transFigure,
        color="#D0D0D0", linewidth=1
    ))

    plt.subplots_adjust(
        left=0.04, right=0.97,
        top=0.75, bottom=0.17, wspace=0.30
    )
    plt.show()

# 8. Main Function
def main():
    image = np.array([
        [10, 10, 10, 10, 10],
        [10, 50, 50, 50, 10],
        [10, 50, 200, 50, 10],
        [10, 50, 50, 50, 10],
        [10, 10, 10, 10, 10]
    ], dtype=float)

    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    conv = convolution2d(image, kernel)
    smooth = smoothing(image)
    sharp = sharpening(image)

    assert all(x.shape == image.shape for x in [conv, smooth, sharp])

    results = [
        ("Input Grayscale", image),
        ("2D Convolution", conv),
        ("Smoothing - Box Filter", smooth),
        ("Sharpening - Laplacian", sharp)
    ]

    print_results(results)
    show_results(results)

if __name__ == "__main__":
    main()
