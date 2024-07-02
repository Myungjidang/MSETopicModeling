import argparse
import easydict

# def get_args(end_date):
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--start_date", type=int, default=2014, help="time series 시작 연도"
#     )
#     parser.add_argument(
#         "--end_date",
#         type=int,
#         default=end_date,
#         help="end_date-1 까지의 time series를 통해 forecasting하는 연도",
#     )
#     parser.add_argument(
#         "--n_cluster",
#         type=int,
#         default=30,
#         help="number of cluster",
#     )
#     parser.add_argument(
#         "--n_word",
#         type=int,
#         default=10,
#         help="number of catch nearest words",
#     )
#     parser.add_argument(
#         "--n_docs",
#         type=int,
#         default=1,
#         help="number of catch nearest docs",
#     )
#     parser.add_argument(
#         "--methods", type=str, default="kmeans", help="clustering method"
#     )
#     parser.add_argument(
#         "--average_years", type=int, default=1, help="최근 n년 연도의 평균과 비교할 때 사용하는 n의 수"
#     )
#     parser.add_argument("--datapath", type=str, default="./sample_data/", help="데이터 경로")
#     parser.add_argument("--save_path", type=str, default="./output", help="Visualization 저장 경로")
#     parser.add_argument(
#         "--n_word_neighbor", type=int, default=20, help="centroid로부터 가까운 이웃단어 개수"
#     )
#     parser.add_argument(
#         "--n_docx_neighbor", type=int, default=3, help="centroid로부터 가까운 이웃문서 개수"
#     )
#     parser.add_argument(
#         "--max_date", type=int, default=2021, help="data가 가지고 있는 최신 연도 정보"
#     )
    
#     args = parser.parse_args()
#     return args



def get_args(end_date):
    args = easydict.EasyDict({
        "start_date" : 2014,
        "end_date" : end_date,
        "n_cluster" : 30,
        "n_word" : 10,
        "n_docs" : 1,
        "methods" : "kmeans",
        "average_years" : 1,
        "datapath" : "./data/",
        "save_path" : "./output",
        "n_word_neighbor" : 20,
        "n_docx_neighbor" : 3,
        "max_date" : 2021
    })
    return args
