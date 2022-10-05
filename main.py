import os.path

from configs.config import Config
from testing.test import test
from training.train import train


def main():
    config_path = os.path.join(os.getcwd(), 'configs', 'config.yaml')
    config = Config(config_path)

    if config.mode == 0:
        train(config)

    elif config.mode == 1:
        test(config)


if __name__ == '__main__':
    main()
