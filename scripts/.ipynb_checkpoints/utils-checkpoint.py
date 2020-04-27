import os
import json
import pandas as pd


def get_next_run_name(log_dir='./logs'):
    try:
        runs = [int(f.split('.')[0][4:]) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f)) if f[:3] == 'run']
        current_run = max(runs) + 1
        return os.path.join(log_dir, 'run_' + str(current_run)) + '.csv'
    except:
        return os.path.join(log_dir, 'run_0') + '.csv'


def write_csv(file_name, runs_df):
    if os.path.exists(file_name):
        old_runs_df = pd.read_csv(file_name, index_col=0)
        runs_df = pd.concat([old_runs_df, runs_df], ignore_index=True, sort=False)
            
    runs_df.to_csv(file_name)


def write_txt(file_name, string):
    with open(file_name, 'a') as logger:
        logger.write(string + '\n')

        
def read_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        return d
