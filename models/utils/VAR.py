import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def forecasting_var(model, train, idx):
    n_fit = 1
    # 가끔 원본데이터 집어넣으면 에러?
    # x.copy한 다음 복사본에 대해 학습하면 
    results = model.fit(n_fit)
    laaged_values = train.values[-n_fit:]
    # forecast = pd.DataFrame(
    #     results.forecast(y=laaged_values, steps=3), index=test.index
    # )
    forecast = pd.DataFrame(
        results.forecast(y=laaged_values, steps=len(idx)), index=idx
    )
    return forecast


def var_dataset(args, mode, measurements):
    if mode == "scala":
        measurements_df = pd.DataFrame(measurements).transpose()
        m_columns = [f"{i}" for i in range(measurements_df.shape[1])]
        measurements_df.columns = m_columns
        return measurements_df
    elif mode == "vector":
        datalist = []
        for i in range(args.n_cluster):
            data = pd.DataFrame(measurements[i])
            datalist.append(data)
        return datalist


def var_scala(scala_measurements, args, test_length):
    test_length = 1
    mydata = var_dataset(args, "scala", scala_measurements)

    if args.end_date > args.max_date:
            train = mydata
    else:
        train = mydata.iloc[:-test_length, :]
        test = mydata.iloc[-test_length:, :]

    forecasting_model = VAR(train)

    if args.end_date > args.max_date:
        start_idx = args.max_date-args.start_date+1
        idx_list = [i for i in range(start_idx, start_idx + (args.end_date - args.max_date))]
        forecast = forecasting_var(forecasting_model, train, idx_list)
        forecast = forecast.transpose()
        idx = list(forecast.columns)[-1]
        var_s_result = forecast.iloc[:,-1].to_frame()
        var_s_result = var_s_result.rename(columns={idx:'predict'})
    else:
        forecast = forecasting_var(forecasting_model, train, [test.index])
        f_n = forecast.to_numpy()[0]
        t_n = test.to_numpy()[0]
        # print(f"VAR {args.end_date} scala  = {np.absolute(f_n - t_n)}")
        print(f"SCALA   | MeanAbsolute      | {np.mean(np.absolute(f_n - t_n))}")   
        forecast = forecast.transpose()
        var_s_result = forecast
    
    var_s_result.columns = ["predict"]
    return var_s_result
    


def var_vector(vector_measurements, args, test_length):
    mydata = var_dataset(args, "vector", vector_measurements)
    distance_list = []
    predict_list = []
    cossim_list = []
    for data in mydata:
        if args.end_date > args.max_date:
            data_length = args.end_date - args.max_date
            train = data.iloc[:-data_length, :]
            forecasting_model = VAR(train)
            start_idx = args.max_date-args.start_date+1
            idx_list = [i for i in range(start_idx, start_idx + (args.end_date - args.max_date))]
            forecast = forecasting_var(forecasting_model, train, idx_list)
            predict_list.append(forecast.values[-1])
        else:
            train = data.iloc[:-test_length, :]
            test = data.iloc[-test_length:, :]
            forecasting_model = VAR(train)
            forecast = forecasting_var(forecasting_model, train, test.index)
            predict_list.append(forecast.values[0])
            distance = np.linalg.norm(forecast - test)
            distance_list.append(distance)
            cossim = cosine_similarity(forecast.values.tolist()[0], test.values.tolist()[0])
            cossim_list.append(cossim)
    

    # 실제값/예측값 비교
    distance_np = np.array(distance_list)
    cossim_np = np.array(cossim_list)
    if args.end_date <= args.max_date:    
        print(f"VECTOR  | EuclideanDistance | {np.mean(distance_np[np.isfinite(distance_np)])}")
        print(f"VECTOR  | CosineSimilarity  | {np.mean(cossim_np[np.isfinite(cossim_np)])}")
    else:
        print("Cannot estimate average score because of forecasting after current years.")
    return predict_list


def var(scala_measurements, vector_measurements, args):
    test_length = 1
    if args.end_date <= args.max_date:
        result_v = var_vector(vector_measurements, args, test_length)
    else:
        result_v = ""
    result_s = var_scala(scala_measurements, args, test_length)
    return result_s, result_v
