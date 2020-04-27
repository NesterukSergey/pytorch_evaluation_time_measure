import time
import platform
import multiprocessing
import pandas as pd
import torch
from torch import nn

from scripts.mobilenet import *
from scripts.utils import *

config = read_json('./config.json')


def get_model(device):
    model = mobilenet_v2()
    model.classifier = nn.Linear(1280, config['num_classes'])
    model.eval()
    model.to(device)
    return model


def test_model_time(model, device, batch_size, iterations):
    model.eval()
    total = iterations * batch_size
    assert total > 0
    
    start_time = time.time()
    for i in range(iterations):
        inp = torch.randn(batch_size, 3, config['input_size'], config['input_size'])
        _ = model(inp.to(device))
        
    return (time.time() - start_time) / total


def main():
    device = "cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu"
    run_file = get_next_run_name(config['log_dir'])
    write_txt(config['log_file'], 'Start new model evaluation')
    
    model_name = 'MobileNetv2'
    model = get_model(device)
    
    for i in [2**i for i in range(config['batch_sizes'])]:
        eval_time = test_model_time(model, device, i, config['iterations'])
        write_txt(config['log_file'], 'Model {} on {}. {} batch size: {}s per image.'.format(model_name, device, i, eval_time))
        run_df = pd.DataFrame({
            'device': device,
            'model_name': model_name,
            'platform': platform.system() + ' ' + platform.release(),
            'proc': platform.processor(),
            'cpus': multiprocessing.cpu_count(),
            'batch_size': i,
            'iterations': config['iterations'],
            'time': eval_time
        }, index=[0])
        write_csv(run_file, run_df)
    
    write_txt(config['log_file'], '--  __  ' * 8)
    write_txt(config['log_file'], '')
    

main()
