This is code that accompanies the paper titled **"Keyword Assisted Embedded Topic Model"** by Bahareh Harandizadeh, J. Hunter Priniski, and Fred Morstatter.
ETM was originally published by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei on a article titled [Topic Modeling in Embedding Spaces](https://arxiv.org/abs/1907.04907). The backbone of this repo has been extracted from the ETM original [code](https://github.com/adjidieng/ETM) and an adoption [code](https://github.com/lffloyd/embedded-topic-model/tree/85a817cf456b02d8ba67ea1f00b984ddf79c68f6) from it. Most of the original code was kept here, with some changes to equip ETM with the ability to incorporate user knowledge in the form of informative topic-level priors over the vocabulary.

# Installation
You can install the package using pip by running: `pip install -U embedded_topic_model`
# Usage
You first need to create a config file and put it in configs folder(a sample for 20newsg dataset is provided). In the configuration file the following items should be set:
```
Dataset:
 - folder-path: the root folder path, all data files(result-file,data-file,sw-file) should be placed in the this path.
 - result-file: the results folder path
 - data-file: input file name
 - sw-file: seed_words file name
Model:
 - bs: batch_size
 - nt: number of topics
 - epochs: number of epochs
 - drop_out: drop_out percentage 
 - theta_act: activation function 
 - lr: learning rate 
 - lambda_theta: Lambda_1 
 - lambda_alpha: Lambda_2 
 - path: The path for save the trained model
```
Then go the the main director(~/embedded-topic-model_mod) and in the terminal running code using:
```
python run.py --config configs/name_config_file.yaml
```

# Citation
To site KeyETM, please use the following link:
```

```
