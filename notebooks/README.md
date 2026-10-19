# Notebooks

The examples of the [post](https://gprietos.github.io/posts/umap/). See the [main README](../README.md#setup) for setup and what each one needs to download.

- [`mnist_umap.ipynb`](mnist_umap.ipynb) ([MNIST Dataset](https://gprietos.github.io/posts/umap/#mnist-dataset)): UMAP of the raw pixels. An interactive 3-D map of the test digits, unsupervised vs. supervised (and semi-supervised) UMAP, metric learning (UMAP fitted on the training digits and applied to the test digits, compared with kNN accuracy), and the post's cover.
- [`openimages_umap.ipynb`](openimages_umap.ipynb) ([Open Images Dataset](https://gprietos.github.io/posts/umap/#open-images-dataset)): a random sample of 25,000 Open Images photos embedded with DINOv2. The map is shown without labels, then colored by groups of classes (animals, food, ...) and by single classes, with k = 50 and k = 15.
- [`umap_explorer.ipynb`](umap_explorer.ipynb) ([Overview](https://gprietos.github.io/posts/umap/#overview), [Effect of k and min_dist](https://gprietos.github.io/posts/umap/#effect-of-k-and-min_dist-hyperparameters)): the 3-D mammoth next to its 2-D UMAP, and the interactive explorer, with several 3-D shapes and sliders for k and min_dist.
- [`unsplash_umap.ipynb`](unsplash_umap.ipynb) (extra, not in the post): the Open Images pipeline on Unsplash Lite photos, colored by their keywords, in 2-D and 3-D. 

On GitHub, the interactive figures appear only as a static preview. To use them, run the notebooks or see them in the post.
