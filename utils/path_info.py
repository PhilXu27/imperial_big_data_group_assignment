from os.path import dirname, abspath, join


root_path = dirname(dirname(abspath(__file__)))
util_path = join(root_path, 'utils')
data_path = join(root_path, 'data')
result_path = join(root_path, 'results')
config_path = join(root_path, 'config')
log_path = join(root_path, 'log')

###############
# Result Path #
###############

raw_data_path = join(data_path, "raw_data")
stock_list_path = join(data_path, "stock_list")
demo_data_path = join(data_path, "demo_data")
main_data_path = join(data_path, "main_data")

###############
# Result Path #
###############
distributed_results = join(result_path, "distributed_results")
baseline_results = join(result_path, "baseline_results")

