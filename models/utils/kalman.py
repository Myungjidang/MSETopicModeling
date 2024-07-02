from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
# from origin import euclidean_distance, cosine_similarity

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def vector(vector_measurements, time_length, mode, args):
    dims = vector_measurements.shape[2]
    idt = np.eye(dims, dtype=np.int8)

    distance_list = []
    cossim_list = []
    result_list = []
    for i, measurements in enumerate(tqdm(vector_measurements)):
        observed_values = measurements
        kf = KalmanFilter(transition_matrices=idt, observation_matrices=idt, transition_covariance=idt * 0.01, observation_covariance=idt * 0.1)
        predicted_values, _ = kf.filter(observed_values)
        smoothed_values, _ = kf.smooth(observed_values)
        
        distance = np.linalg.norm(predicted_values[-1, :] - measurements[-1])
        cossim = cosine_similarity(predicted_values[-1, :], measurements[-1])

        distance_list.append(distance)
        cossim_list.append(cossim)
        result_list.append(predicted_values[-1, :])
    result_e = np.mean(np.array(distance_list))
    result_c = np.mean(np.array(cossim_list))
    print(f"VECTOR  | EuclideanDistance | {result_e}")
    print(f"VECTOR  | CosineSimilarity  | {result_c}")
    return np.array(result_list)

def scala(scala_measurements, time_length, mode, args):  # 구현체
    kalman_scala_result = []
    result = []
    for idx, scala_measurement in enumerate(scala_measurements):
        observed_values = scala_measurement
        kf = KalmanFilter(transition_matrices=1, observation_matrices=1, transition_covariance=0.1, observation_covariance=0.1)

        # Filter the data using the Kalman filter
        predicted_values, _ = kf.filter(observed_values.reshape(-1,1))

        # Apply smoothing to the filtered data
        smoothed_values, _ = kf.smooth(observed_values.reshape(-1,1))

        kalman_scala_result.append(round(predicted_values[-1][0], 4))
        result.append(abs(round(predicted_values[-1][0], 4)-scala_measurements[idx][-1]))
    print(f"SCALA   | MeanAbsolute      | {np.mean(np.array(result))}")
    return result

def remake_shape(vector_measurements, time_length, args):
    """
    this function reshape (topic, timeline, shape) to (timeline, topic, shape)
    input : vector_measurements, time_length, args
    output : new shape of measurements
    """
    new_measurements = [[] for i in range(time_length)]

    for topic_measurements in vector_measurements:
        for i, time_measurements in enumerate(topic_measurements):
            new_measurements[i].append(list(time_measurements))

    return np.array(new_measurements)

def kalmanfilter(scala_measurements, vector_measurements, args, mode):
    time_length = args.end_date - args.start_date + 1
    kalman_s = scala(scala_measurements, time_length, mode, args)
    kalman_v = vector(vector_measurements, time_length, mode, args)
    
    return kalman_s, kalman_v

if __name__ == "__main__":
    kalmanfilter()