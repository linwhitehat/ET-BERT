#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import os,shutil
import json
import random

def combine_data_json():
    target_path = "F:\\dataset\\"
    target_file1 = target_path + "dataset_1.json"
    target_file2 = target_path + "dataset_2.json"
    save_file = target_path + "dataset.json"

    result_samples = {'0':{'payload':{}}, '1':{'payload':{}}}
    with open(target_file1, 'r', encoding='utf-8') as f:
        file1_json = json.load(f)
    with open(target_file2, 'r', encoding='utf-8') as f:
        file2_json = json.load(f)

    count = 0
    for key in file1_json.keys():
        for item_key in file1_json[key]['payload'].keys():
            if len(file1_json[key]['payload'][item_key]) > 100:
                count += 1
                result_samples['0']['payload'][str(count)] = file1_json[key]['payload'][item_key]
    file1_random_samples = random.sample(result_samples['0']['payload'].keys(), 5000)

    count = 0
    for key in file2_json.keys():
        for item_key in file2_json[key]['payload'].keys():
            if len(file2_json[key]['payload'][item_key]) > 100:
                count += 1
                result_samples['1']['payload'][str(count)] = file2_json[key]['payload'][item_key]
    file2_random_samples = random.sample(result_samples['1']['payload'].keys(), 5000)

    combined_result = {'0':{'samples':len(file1_random_samples), 'payload':{}}, '1':{'samples':len(file2_random_samples), 'payload':{}}}
    count = 0
    for i in file1_random_samples:
        count += 1
        combined_result['0']['payload'][str(count)] = result_samples['0']['payload'][i]
    count = 0
    for i in file2_random_samples:
        count += 1
        combined_result['1']['payload'][str(count)] = result_samples['1']['payload'][i]

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(combined_result,f,indent=4,ensure_ascii=False)
    return 0

def basic_process_1(y_dict,x_dict):
    x_len_train_new = []
    x_len_test_new = []
    x_len_valid_new = []
    x_time_train_new = []
    x_time_test_new = []
    x_time_valid_new = []

    bind_data = [y_dict['train'], y_dict['test'], y_dict['valid']]
    bind_len_data = [x_dict['len_train'], x_dict['len_test'], x_dict['len_valid']]
    bind_len_data_new = [x_len_train_new, x_len_test_new, x_len_valid_new]
    bind_time_data = [x_dict['time_train'], x_dict['time_test'], x_dict['time_valid']]
    bind_time_data_new = [x_time_train_new, x_time_test_new, x_time_valid_new]
    
    for data_index in range(len(bind_data)):
        for item_index in range(len(bind_data[data_index])):
            
            i = 0
            
            len_data = bind_len_data[data_index][item_index].tolist()
            time_data = bind_time_data[data_index][item_index].tolist()
            length = len(len_data)
            while i < length:
                if len_data[i] == 0:
                    len_data.remove(0)
                    if 0 in time_data:
                        time_data.remove(0)
                    i -= 1
                    length -= 1
                i += 1
            
            time_data_new = time_process_1(time_data)
            
            bind_len_data_new[data_index].append(len_data)
            bind_time_data_new[data_index].append(time_data_new)
    
    x_len_train_np = np.array(x_len_train_new)
    x_len_test_np = np.array(x_len_test_new)
    x_len_valid_np = np.array(x_len_valid_new)
    x_time_train_np = np.array(x_time_train_new)
    x_time_test_np = np.array(x_time_test_new)
    x_time_valid_np = np.array(x_time_valid_new)

    x_result = {'len_train':x_len_train_np,'len_test':x_len_test_np,'len_valid':x_len_valid_np,'time_train':x_time_train_np,'time_test':x_time_test_np,'time_valid':x_time_valid_np}
    return x_result

def time_process_1(time_data):
    
    time_data_new = [0.0]
    time_data_new.extend([time_data[time_index] - time_data[time_index - 1] for time_index in range(1, len(time_data))])
    return time_data_new

if __name__ == '__main__':
    combine_data_json()
