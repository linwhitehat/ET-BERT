#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import os,shutil
import json
import random

def combine_data_json():
    target_path = "F:\\LEFTT-2021\\binary\\"
    target_file1 = target_path + "ssr_header_payload_2_dataset.json"
    target_file2 = target_path + "vmess_header_payload_dataset.json"
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

def copy_dir_tls13_data():
    label_name_list = ['163.com', '51.la', '51cto.com', 'acm.org', 'adobe.com', 'alibaba.com', 'alicdn.com', 'alipay.com', 'amap.com', 'amazonaws.com', 'ampproject.org', 'apple.com', 'arxiv.org', 'asus.com', 'atlassian.net', 'azureedge.net', 'baidu.com', 'bilibili.com', 'biligame.com', 'booking.com', 'chia.net', 'chinatax.gov.cn', 'cisco.com', 'cloudflare.com', 'cloudfront.net', 'cnblogs.com', 'codepen.io', 'crazyegg.com', 'criteo.com', 'ctrip.com', 'dailymotion.com', 'deepl.com', 'digitaloceanspaces.com', 'duckduckgo.com', 'eastday.com', 'eastmoney.com', 'elsevier.com', 'facebook.com', 'feishu.cn', 'ggpht.com', 'github.com', 'gitlab.com', 'gmail.com', 'goat.com', 'google.com', 'grammarly.com', 'gravatar.com', 'guancha.cn', 'huanqiu.com', 'huawei.com', 'hubspot.com', 'huya.com', 'ibm.com', 'icloud.com', 'ieee.org', 'instagram.com', 'iqiyi.com', 'jb51.net', 'jd.com', 'kugou.com', 'leetcode-cn.com', 'media.net', 'mi.com', 'microsoft.com', 'mozilla.org', 'msn.com', 'naver.com', 'netflix.com', 'nike.com', 'notion.so', 'nvidia.com', 'office.net', 'onlinedown.net', 'opera.com', 'oracle.com', 'outbrain.com', 'overleaf.com', 'paypal.com', 'pinduoduo.com', 'python.org', 'qcloud.com', 'qq.com', 'researchgate.net', 'runoob.com', 'sciencedirect.com', 'semanticscholar.org', 'sina.com.cn', 'smzdm.com', 'snapchat.com', 'sohu.com', 'spring.io', 'springer.com', 'squarespace.com', 'statcounter.com', 'steampowered.com', 't.co', 'taboola.com', 'teads.tv', 'thepaper.cn', 'tiktok.com', 'toutiao.com', 'twimg.com', 'twitter.com', 'unity3d.com', 'v2ex.com', 'vivo.com.cn', 'vk.com', 'vmware.com', 'walmart.com', 'weibo.com', 'wikimedia.org', 'wikipedia.org', 'wp.com', 'xiaomi.com', 'ximalaya.com', 'yahoo.com', 'yandex.ru', 'youtube.com', 'yy.com', 'zhihu.com']
    
    root_path = "I:\\cstnet-tls1.3\\labeled\\"
    
    target_path = "K:\\cstnet-tls1.3\\120\\"
    
    for label_index in range(len(label_name_list)):
        src_dir = root_path + label_name_list[label_index]
        dst_dir  = target_path + label_name_list[label_index]
        print("copying %s from %s to %s .." % (label_name_list[label_index], src_dir, dst_dir))
        shutil.copytree(src=src_dir, dst=dst_dir)
    return 0

def copy_record_tls13_data():
    target_path = "K:\\cstnet-tls1.3\\"
    source_path_file = "I:\\cstnet-tls1.3\\picked_file_record"
    with open(source_path_file,"r") as f:
        records = f.read().split("\n")
    temp_label = ""
    count = 0
    for record_index in range(len(records)):
        src_file = records[record_index]
        label = src_file.split("\\")[6]
        
        if temp_label != label:
            temp_label = label
            count = 1
        else:
            count += 1
        #file = src_file.split("\\")[-1]
        dst_file = target_path + label + "\\" + str(count) + ".pcap"
        if not os.path.exists(target_path + label):
            os.mkdir(target_path + label)
        shutil.copy(src=src_file,dst=dst_file)
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
