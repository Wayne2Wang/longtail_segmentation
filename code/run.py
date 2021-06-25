import torch
import yaml

from main import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print('Start Training...')
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
