import os
import pandas as pd


def get_text_data(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    

def read_files_in_directory(directory):
    files = get_files_in_directory(directory)
    data = {}
    for file in files:
        file_path = os.path.join(directory, file)
        data[os.path.abspath(file_path)] = get_text_data(file_path)
    return data


def get_files_in_directory(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':

    dataset_path = r'./dataset'
    files = get_files_in_directory(dataset_path)
    data = read_files_in_directory(dataset_path)

    df = pd.DataFrame(data.items(), columns=['file_path', 'text'])

    df.to_csv('dataset.csv', index=False)
    