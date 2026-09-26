import numpy as np
def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    points = np.array(points)
    assignments = np.array(assignments)
    list1 = []
    sum1 = [0] * k
    intcheck = [0] * k
    for i in range(len(points)):
        check = assignments[i]
        sum1[check] = sum1[check] + points[i]
        intcheck[check] = intcheck[check] + 1

    for i in range(k):
        if intcheck[i] != 0:
            answer = sum1[i] / intcheck[i]
            list1.append(answer.tolist())
        else:
            list1.append([0, 0])
    return list1 