# Library
import numpy as np
import pandas as pd
# import sys

# Main process
from .utils.arguments import get_args
from .utils.average import average
from .utils.labels import label_centroid
from .utils.nearest import word_nearest, word_nearest_onecountry
from .utils.nearest import docx_nearest, docx_nearest_onecountry
from .utils.preprocess import make_base, docx_number, first_excute, make_docx_v, datasave
from .utils.VAR import var
from .utils.kalman import kalmanfilter

# Visualize
from .utils.visualization import visual_scala_total 
from .utils.visualization import visual_scala_onescore, visual_scala_relative
from .utils.visualization import visual_vector


def patent_prediction(country_list, args):
    docx_v_df = pd.DataFrame(make_docx_v(args, country_list))
    label, centroids = label_centroid(docx_v_df, args.n_cluster) # path 여기까지 변경

    label_df = pd.DataFrame(label, columns=["label"])
    (
        scala_measurements,
        scala_motion_controls,
        vector_measurements,
        vector_motion_controls
    ) = make_base(args, label, centroids)

    # print(scala_measurements, vector_measurements)
    print(f"[{args.end_date}]")
    print("-----------------Result of timeseries forecasting-----------------")
    print("[ VAR ]")    
    # VAR
    var_s, var_v = var(scala_measurements, vector_measurements, args)
    datasave(var_s, args.save_path, "var", "scala", args)
    datasave(var_v, args.save_path, "var", "vector", args)
    

    # Average
    for i in [1,3,5]:
        print("------------------------------------------------------------------")
        print(f"[ Average : {i} years ]")
        if args.end_date <= args.max_date:
            average(args, scala_measurements, i, "scala")
            average(args, vector_measurements, i, "vector")
        else:
            print("Cannot estimate average score because of forecasting after current years.")
    print(" ")
    return var_v

if __name__ == "__main__":
    country_list = ["cn", "de", "en", "jp", "kr"] # 다국어 리스트
    args = get_args()
    first_excute(args.save_path)
    patent_prediction(country_list, args)
