# Library
import numpy as np
import pandas as pd
import sys

# Main process
from .utils.arguments import get_args
from .utils.labels import label_centroid
from .utils.nearest import word_nearest, word_nearest_onecountry
from .utils.nearest import docx_nearest
from .utils.preprocess import make_base, docx_number, first_excute, make_docx_v, datasave
from .utils.VAR import var
from .utils.kalman import kalmanfilter

# Visualize
from .utils.visualization import visual_scala_total 
from .utils.visualization import visual_scala_onescore, visual_scala_relative
from .utils.visualization import visual_vector

def visualization(country_list, args):
    docx_v_df = pd.DataFrame(make_docx_v(args, country_list))
    label, centroids = label_centroid(docx_v_df, args.n_cluster) # path 여기까지 변경

    label_df = pd.DataFrame(label, columns=["label"])
    (
        scala_measurements,
        scala_motion_controls,
        vector_measurements,
        vector_motion_controls
    ) = make_base(args, label, centroids)

    var_s, var_v = var(scala_measurements, vector_measurements, args)
    # Visualization 
    docx_num_list = docx_number(args, label, centroids)
    visual_scala_total(args, scala_measurements)
    visual_scala_onescore(args, scala_measurements)
    visual_scala_relative(args, scala_measurements, docx_num_list)

    visual_vector(args, vector_measurements, centroids, "var")

    # Nearest (Total)
    docx_nearest(args, centroids, label_df, docx_v_df, country_list, args.n_docx_neighbor, "total")
    word_nearest(args, centroids, args.n_word_neighbor, country_list, "total")
    word_nearest_onecountry(args, centroids, args.n_word_neighbor, country_list, "total")

    # Nearest (VAR)
    docx_nearest(args, var_v, label_df, docx_v_df, country_list, args.n_docx_neighbor, "var")
    word_nearest(args, var_v, args.n_word_neighbor, country_list, "var")
    word_nearest_onecountry(args, var_v, args.n_word_neighbor, country_list, "var")

    # # Save data
    datasave(scala_measurements, args.save_path, "measurements", "scala", args)
    datasave(vector_measurements, args.save_path, "measurements", "vector", args)


if __name__ == "__main__":
    # country_list = ["cn", "de", "en", "jp", "kr"] # 다국어 리스트
    country_list = ["chinese", "english", "german", "japanese", "korean"]
    current_years = 2023
    args = get_args(current_years)
    first_excute(args.save_path)
    visualization(country_list, args)
