import pickle
import configparser

def load_pkl_file(file_path):
    """
    读取并加载 .pkl 文件中的数据

    参数:
    file_path (str): .pkl 文件的路径

    返回:
    object: 从 .pkl 文件中加载的数据
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def recreate(file_path, Dataset, Task):

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(file_path)

    config.set('DataSet', 'DATA_SET', str(Dataset))
    config.set('DataSet', 'Task', str(Task))

    with open(file_path, 'w') as configfile:
        config.write(configfile)



if __name__ == '__main__':

    Dataset = 'TADPOLE'
    Task = 'SMCI_PMCI'

    pkl_file_path = 'config_opt.pkl'
    data = load_pkl_file(pkl_file_path)

    for file_path in data:
        recreate(file_path, Dataset, Task)