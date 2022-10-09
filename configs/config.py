import yaml


class Config:

    def __init__(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))
        self.train_path = config['data']['train_path']
        self.test_path = config['data']['test_path']
        self.batch_size = config['data']['batch_size']
        self.pickle_data = config['data']['pickle_data']

        self.epochs = config['train']['epochs']
        self.lr = float(config['train']['lr'])
        self.validation_split = config['train']['validation_split']
        self.balance_mode = config['train']['balance_mode']

        self.save_mode = config['model']['save_mode']
        self.save_path = config['model']['save_path']
        self.load_train = config['model']['load_train']
        self.load_path = config['model']['load_path']
        self.mode = config['model']['model_mode']
        self.num_outputs = config['model']['num_outputs']

