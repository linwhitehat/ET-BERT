## Description of processing PCAP files to generate dataset
For PCAP data, it is recommended to clean it first. Since the program processing logic is not smooth, we detail the data pre-processing for pre-training and fine-tuning as followed.

### Pre-training Stage
*Main Program*: dataset_generation.py

*Functions*: pretrain_dataset_generation, get_burst_feature

1. Initialization. 
Set the variable `pcap_path` (line:616) as the directory of PCAP data to be processed. 
Set the variable `word_dir` (line:23) and `word_name` (line:24) as the storage directory of pre-training daraset.

2. Pre-process PCAP. 
Set the variable `output_split_path` (line:583) and `pcap_output_path` (line:584). 
The `pcap_output_path` indicates the storage directory where the pcapng format of PCAP data is converted to pcap format. 
The `output_split_path` represents the storage directory for PCAP data slicing into session format. 

3. Gnerate Pre-training Datasets. 
Following the completion of PCAP data processing, the program generates a pre-training dataset composed of BURST.

### Fine-tuning Stage
*Main Program*: main.py

*Functions*: data_preprocess.py, dataset_generation.py, open_dataset_deal.py, dataset_cleanning.py

The key idea of the fine-tuning phase when processing public PCAP datasets is to first distinguish folders for different labeled data in the dataset, then perform session slicing on the data, and finally generate packet-level or flow-level datasets according to sample needs.

**Note:** Due to the complexity of the possible existence of raw PCAP data, it is recommended that the following steps be performed to check the code execution when it reports an error.

1. Initialization. 
`pcap_path`, `dataset_save_path`, `samples`, `features`, `dataset_level` (line:28) are the basis variables, which represent the original data directory, the stored generated data directory, the number of samples, the feature type, and the data level. `open_dataset_not_pcap` (line:215)  represents the processing of converting PCAP data to pcap format, e.g. pcapng to pcap. 
And `file2dir` (line:226) represents the generation of category directories to store PCAP data when a pcap file is a category. 

2. Pre-process. 
The data pre-processing is primarily to split the PCAP data in the directory into session data. 
Please set the `splitcap_finish` parameter to 0 to initialize the sample number array, and the value of `sample` set at this time should not exceed the minimum number of samples. 
Then you can set `splitcap=True` (line:54) and run the code for splitting PCAP data. The splitted sessions will be saved in `pcap_path\\splitcap`.

3. Generation. 
After data pre-processing is completed, variables need to be changed for generating fine-tuned training data. The `pcap_path` should be the path of splitted data and set 
`splitcap=False`. Now the `sample` can be unrestricted by the minimum sample size. The `open_dataset_not_pcap` and `file2dir` should be False. Then the dataset for fine-tuning will be generated and saved in `dataset_save_path`. 
