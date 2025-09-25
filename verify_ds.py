import pickle

# Path to the pickle file
pickle_file = 'data.pickle'

try:
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Number of samples: {len(data_dict['data'])}")
    print(f"Labels: {set(data_dict['labels'])}")
except FileNotFoundError:
    print(f"File {pickle_file} not found. Please ensure create_dataset.py ran successfully.")
except Exception as e:
    print(f"Error reading {pickle_file}: {e}")