import numpy as np
import pandas as pd

def euclidean_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        pt1[i] = float(pt1[i])
        pt2[i] = float(pt2[i])
        distance += (pt1[i] - pt2[i]) ** 2
    return distance**0.5


def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def average(args, measurements, years, mode):
    average_list = []
    result = []
    if mode == "scala":
        for i, measurement in enumerate(measurements):
            actual = measurement[-1]
            measurement = measurement[-years - 1 : -1]
            measurement_mean = sum(measurement) / years
            average_list.append(measurement_mean)
            result.append(abs(measurement_mean - actual))
        average_list = pd.DataFrame(average_list, columns=["predict"])
        average_list.to_csv(f"{args.save_path}/average/scala/{args.end_date}_{years}.csv", index=False)
        # result = average_list.mean(axis="rows")
        result = np.array(result)
        print(f"SCALA   | MeanAbsolute      | {np.mean(result)}")
        
    else:
        mean_vector_list = []
        cossim_list = []
        for i, measurement in enumerate(measurements):
            centroid = np.array(measurement[-1])
            measurement = measurement[-years - 1 : -1]
            mean = np.mean(measurement, axis=0)
            result = euclidean_distance(mean, centroid)
            average_list.append(result)
            cossim = cosine_similarity(mean, centroid)
            cossim_list.append(cossim)
            mean_vector_list.append(mean)
        average_list = pd.DataFrame(average_list, columns=["predict"])
        average_list.to_csv(f"{args.save_path}/average/vector/euclidean_distance_{args.end_date}_{years}.csv", index=False)
        mean_vector_list = pd.DataFrame(mean_vector_list)
        
        result = average_list.mean(axis="rows").to_numpy()[0] 
        print(f"VECTOR  | EuclideanDistance | {result}")
        print(f"VECTOR  | CosineSimilarity  | {np.mean(np.array(cossim_list))}")

