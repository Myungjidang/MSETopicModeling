import os
import sys
from .main import patent_prediction
from .visualization import visualization
from .utils.arguments import get_args
from .utils.preprocess import first_excute

def execute_patent(current_years, path):
    country_list = ["chinese", "english", "german", "japanese", "korean"]
    args = get_args(current_years)
    first_excute(args.save_path)
    file_name = f"{path}/output/results_{2017}_to_{current_years}.txt"

    # sys.stdout = open(file_name, 'w')
    for i in range(2017, current_years+1):
        patent_args = get_args(i)
        var_v = patent_prediction(country_list, patent_args)
    return var_v 


if __name__ == "__main__":
    current_years = 2019
    path = os.getcwd()
    var_v = execute_patent(current_years, path)
    country_list = ["chinese", "english", "german", "japanese", "korean"]