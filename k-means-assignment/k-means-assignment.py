import numpy as np
def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    points = np.array(points)
    centroids = np.array(centroids)
    list1 = []
    for i in range(len(points)):
        min = np.linalg.norm(points[i] - centroids[0])
        check = 0
        for j in range(len(centroids)):
            b = np.linalg.norm(points[i] - centroids[j])
            if b < min:
                min = b
                check = j
                
        list1.append(check)

    return list1