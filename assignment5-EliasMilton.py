'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


DATASETS = ("dataset1.npy", "dataset2.npy")
IMAGE_DIR = Path("images")
MIN_SAMPLES = 5


#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()


def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()


def get_ground_level(pcd, bins=80, return_histogram=False):
    """
    Estimate the ground level used for filtering from the z-values.

    The real ground surface appears as a large low-z peak. The returned value is
    placed at the first clear valley after that peak, so it removes the whole
    dense ground band rather than only marking the center height of the ground.
    """
    z_values = pcd[:, 2]
    counts, bin_edges = np.histogram(z_values, bins=bins)

    ground_peak_index = int(np.argmax(counts))
    peak_count = counts[ground_peak_index]
    valley_threshold = peak_count * 0.05

    valley_index = None
    for index in range(ground_peak_index + 1, len(counts)):
        if counts[index] <= valley_threshold:
            valley_index = index
            break

    if valley_index is None:
        valley_index = ground_peak_index

    ground_level = float(bin_edges[valley_index + 1])

    if return_histogram:
        return ground_level, counts, bin_edges, ground_peak_index, valley_index

    return ground_level


#%% read file containing point cloud data
pcd = np.load("dataset1.npy")

pcd.shape

#%% show downsampled data in external window
#%matplotlib qt
#show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
def plot_ground_histogram(dataset_name, pcd, output_dir=IMAGE_DIR, show=False):
    """Create the z histogram used to estimate ground level and save it as PNG."""
    output_dir.mkdir(exist_ok=True)
    ground_level, counts, bin_edges, peak_index, valley_index = get_ground_level(
        pcd, return_histogram=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pcd[:, 2], bins=bin_edges, color="#4c78a8", edgecolor="white")
    ax.axvline(
        ground_level,
        color="#e45756",
        linestyle="--",
        linewidth=2,
        label=f"Ground level used for filtering = {ground_level:.2f}",
    )
    ax.set_title(f"Z histogram for {dataset_name}")
    ax.set_xlabel("z value")
    ax.set_ylabel("Number of points")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{Path(dataset_name).stem}_ground_histogram.png"
    fig.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "dataset": dataset_name,
        "ground_level": ground_level,
        "histogram_path": output_path,
        "ground_peak_bin": (float(bin_edges[peak_index]), float(bin_edges[peak_index + 1])),
        "valley_bin": (float(bin_edges[valley_index]), float(bin_edges[valley_index + 1])),
    }


est_ground_level = get_ground_level(pcd)
print(est_ground_level)

pcd_above_ground = pcd[pcd[:,2] > est_ground_level]
#%%
pcd_above_ground.shape

#%% side view
#show_cloud(pcd_above_ground)


# %%
unoptimal_eps = 10
# find the elbow
# The original value is kept as a reference from the template. The optimized
# value is calculated below with the k-distance elbow method.
#clustering = DBSCAN(eps = unoptimal_eps, min_samples=5).fit(pcd_above_ground)

#%%
#clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
#colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

# %%
# Plotting resulting clusters
#plt.figure(figsize=(10,10))
#plt.scatter(pcd_above_ground[:,0],
#            pcd_above_ground[:,1],
#            c=clustering.labels_,
#            cmap=matplotlib.colors.ListedColormap(colors),
#            s=2)
#
#
#plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
#plt.xlabel('x axis',fontsize=14)
#plt.ylabel('y axis',fontsize=14)
#plt.show()


#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''
def estimate_eps(points, min_samples=MIN_SAMPLES):
    """
    Estimate DBSCAN eps from the elbow in the sorted k-distance curve.

    The knee is selected as the point with the largest perpendicular distance
    from the straight line between the first and last sorted k-distance values.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(points)
    distances, _ = neighbors.kneighbors(points)
    k_distances = np.sort(distances[:, min_samples - 1])

    x = np.arange(k_distances.size)
    start = np.array([x[0], k_distances[0]])
    end = np.array([x[-1], k_distances[-1]])
    line = end - start
    line_length = np.linalg.norm(line)

    if line_length == 0:
        return float(k_distances[-1]), k_distances, 0

    distances_to_line = np.abs(
        line[0] * (start[1] - k_distances) - line[1] * (start[0] - x)
    ) / line_length
    knee_index = int(np.argmax(distances_to_line))
    eps = float(k_distances[knee_index])
    return eps, k_distances, knee_index


def plot_elbow(dataset_name, k_distances, knee_index, eps, output_dir=IMAGE_DIR, show=False):
    output_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_distances, color="#4c78a8", linewidth=1.5)
    ax.axvline(knee_index, color="#e45756", linestyle="--", label=f"Estimated eps = {eps:.3f}")
    ax.axhline(eps, color="#e45756", linestyle=":")
    ax.set_title(f"k-distance elbow plot for {dataset_name}")
    ax.set_xlabel("Points sorted by 5th-nearest-neighbor distance")
    ax.set_ylabel("5th-nearest-neighbor distance")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{Path(dataset_name).stem}_elbow_plot.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def cluster_points(points, eps, min_samples=MIN_SAMPLES):
    """Run DBSCAN and return cluster labels."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)


def plot_clusters(dataset_name, points, labels, output_dir=IMAGE_DIR, show=False):
    output_dir.mkdir(exist_ok=True)
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(clusters, 1))]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap=matplotlib.colors.ListedColormap(colors),
        s=2,
    )
    ax.set_title("DBSCAN: %d clusters" % clusters, fontsize=20)
    ax.set_xlabel("x axis", fontsize=14)
    ax.set_ylabel("y axis", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    output_path = output_dir / f"{Path(dataset_name).stem}_dbscan_clusters.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
def find_catenary_cluster(points, labels):
    """
    Find the likely catenary cluster.

    Noise is labelled -1 by DBSCAN and is ignored. The selected cluster is the
    non-noise cluster with the largest combined x/y span.
    """
    best_label = None
    best_span = -1.0
    best_bounds = None

    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster = points[labels == label]
        min_x, min_y = cluster[:, 0].min(), cluster[:, 1].min()
        max_x, max_y = cluster[:, 0].max(), cluster[:, 1].max()
        span = (max_x - min_x) + (max_y - min_y)

        if span > best_span:
            best_label = int(label)
            best_span = float(span)
            best_bounds = {
                "min_x": float(min_x),
                "min_y": float(min_y),
                "max_x": float(max_x),
                "max_y": float(max_y),
                "xy_span": float(span),
                "point_count": int(cluster.shape[0]),
            }

    return best_label, best_bounds


def plot_catenary(dataset_name, points, labels, catenary_label, output_dir=IMAGE_DIR, show=False):
    output_dir.mkdir(exist_ok=True)
    catenary = points[labels == catenary_label]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], color="lightgray", s=1, label="Other above-ground points")
    ax.scatter(catenary[:, 0], catenary[:, 1], color="#e45756", s=3, label="Selected catenary cluster")
    ax.set_title(f"Selected catenary cluster for {dataset_name}")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=4)
    fig.tight_layout()

    output_path = output_dir / f"{Path(dataset_name).stem}_catenary_cluster.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def process_dataset(dataset_name):
    """Run Tasks 1-3 for one dataset."""
    pcd = np.load(dataset_name)
    ground_result = plot_ground_histogram(dataset_name, pcd)
    ground_level = ground_result["ground_level"]
    pcd_above_ground = pcd[pcd[:, 2] > ground_level]

    eps, k_distances, knee_index = estimate_eps(pcd_above_ground)
    elbow_path = plot_elbow(dataset_name, k_distances, knee_index, eps)
    labels = cluster_points(pcd_above_ground, eps)
    clusters_path = plot_clusters(dataset_name, pcd_above_ground, labels)
    catenary_label, catenary_bounds = find_catenary_cluster(pcd_above_ground, labels)
    catenary_path = plot_catenary(dataset_name, pcd_above_ground, labels, catenary_label)

    return {
        "dataset": dataset_name,
        "ground_level": ground_level,
        "total_points": int(pcd.shape[0]),
        "points_above_ground": int(pcd_above_ground.shape[0]),
        "eps": eps,
        "cluster_count": len(set(labels)) - (1 if -1 in labels else 0),
        "noise_points": int(np.sum(labels == -1)),
        "catenary_label": catenary_label,
        "catenary_bounds": catenary_bounds,
        "histogram_path": ground_result["histogram_path"],
        "elbow_path": elbow_path,
        "clusters_path": clusters_path,
        "catenary_path": catenary_path,
    }


def main():
    results = [process_dataset(dataset_name) for dataset_name in DATASETS]

    for result in results:
        print(f"{result['dataset']}")
        print(f"  Ground level used for filtering: {result['ground_level']:.2f}")
        print(f"  Optimal eps: {result['eps']:.3f}")
        print(f"  Clusters: {result['cluster_count']}")
        print(f"  Noise points: {result['noise_points']}")
        print(f"  Catenary cluster label: {result['catenary_label']}")
        print(f"  Catenary bounds: {result['catenary_bounds']}")
        print(f"  Histogram plot: {result['histogram_path']}")
        print(f"  Elbow plot: {result['elbow_path']}")
        print(f"  Cluster plot: {result['clusters_path']}")
        print(f"  Catenary plot: {result['catenary_path']}")


if __name__ == "__main__":
    main()
