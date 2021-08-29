import os
import pickle
import json
import pandas as pd
import numpy as np

def gene_config(log_dir):

    for root, _, files in os.walk(log_dir):
        if 'progress.csv' in files:
            print(root)
            config_name = os.path.join(root, 'config.json')
            env_name = root.split('/')[-1].split('_')[-3]
            exp_name = '_'.join(['SIL', env_name])
            seed = root.split('/')[-1].split('_')[-1]
            with open(config_name, 'w') as  config:
                config_dict = dict(exp_name=exp_name, args=dict(seed=seed))
                json_data = json.dumps(config_dict)
                config.write(json_data)


def get_datasets(log_dir):

    for root, _, files in os.walk(log_dir):
        if 'progress.csv' in files:
            print(root, end=' ===> ')

            file_name = os.path.join(root, 'progress.csv')
            out_name = os.path.join(root, 'progress.txt')


            # import ipdb;ipdb.set_trace()
            data = pd.read_csv(file_name)
            data.rename(columns = {"serial_timesteps": "TotalEnvInteracts", "eprewmean":"AverageTestEpRet"},  inplace=True)
            clip_data = data[['TotalEnvInteracts', 'AverageTestEpRet']]
            last_rew = clip_data['AverageTestEpRet'].iloc[-1]
            clip_data = clip_data.append([{'TotalEnvInteracts':1100000,'AverageTestEpRet':last_rew}], ignore_index=True)
            clip_data.to_csv(out_name, sep='\t',index=False)

if __name__ == '__main__':
    log_dir = os.path.join(os.getcwd(), 'tmp')
    gene_config(log_dir)
    get_datasets(log_dir)
