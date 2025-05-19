import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio
import seaborn as sns

def read_image(file_path):
    with rasterio.open(file_path) as src:
        if src.count < 4:
            raise ValueError(f"Image must have at least 4 bands, found: {src.count}")
        blue = src.read(1).astype(float)
        green = src.read(2).astype(float)
        red = src.read(3).astype(float)
        nir = src.read(4).astype(float)
    return red, green, blue, nir

def prepare_data(red, green, blue, nir):
    stacked = np.stack([red, green, blue, nir], axis=-1)
    pixels = stacked.reshape((-1, 4))
    return pixels, red.shape

def apply_kmeans(pixels, rows, cols, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape((rows, cols))

def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-5)
    return np.clip(ndvi, -1, 1)

def calculate_correlation(red, green, blue, nir):
    flat_data = [red.flatten(), green.flatten(), blue.flatten(), nir.flatten()]
    return np.corrcoef(flat_data)

def create_rgb(red, green, blue):
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_norm = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    return rgb_norm

def visualize_results(red, green, blue, nir, rgb, clustered, ndvi, correlation):
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))

    im = axs[0, 0].imshow(red, cmap='Reds')
    axs[0, 0].set_title('Red Band')
    axs[0, 0].axis('off')
    plt.colorbar(im, ax=axs[0, 0])

    im = axs[0, 1].imshow(green, cmap='Greens')
    axs[0, 1].set_title('Green Band')
    axs[0, 1].axis('off')
    plt.colorbar(im, ax=axs[0, 1])

    im = axs[0, 2].imshow(blue, cmap='Blues')
    axs[0, 2].set_title('Blue Band')
    axs[0, 2].axis('off')
    plt.colorbar(im, ax=axs[0, 2])

    im = axs[0, 3].imshow(nir, cmap='gray')
    axs[0, 3].set_title('NIR Band')
    axs[0, 3].axis('off')
    plt.colorbar(im, ax=axs[0, 3])

    axs[1, 0].imshow(rgb)
    axs[1, 0].set_title('RGB Combined')
    axs[1, 0].axis('off')

    im = axs[1, 1].imshow(clustered, cmap='tab10')
    axs[1, 1].set_title('K-Means Clustering (k=5)')
    axs[1, 1].axis('off')
    plt.colorbar(im, ax=axs[1, 1])

    sns.heatmap(correlation, annot=True, xticklabels=['Red', 'Green', 'Blue', 'NIR'],
                yticklabels=['Red', 'Green', 'Blue', 'NIR'], cmap='coolwarm', ax=axs[1, 2])
    axs[1, 2].set_title('Correlation Matrix')

    im = axs[1, 3].imshow(ndvi, cmap='RdYlGn')
    axs[1, 3].set_title('NDVI')
    axs[1, 3].axis('off')
    plt.colorbar(im, ax=axs[1, 3])

    plt.tight_layout()
    plt.show()

def main():
    file_path = "multispectral.tif"
    try:
        red, green, blue, nir = read_image(file_path)
        pixels, (rows, cols) = prepare_data(red, green, blue, nir)
        clustered = apply_kmeans(pixels, rows, cols)
        ndvi = calculate_ndvi(red, nir)
        correlation = calculate_correlation(red, green, blue, nir)
        rgb = create_rgb(red, green, blue)
        visualize_results(red, green, blue, nir, rgb, clustered, ndvi, correlation)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()