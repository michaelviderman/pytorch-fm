## The code was forked and modified from https://github.com/rixwew/pytorch-fm, written by rixwew. The API Documentation of the original code is: https://rixwew.github.io/pytorch-fm

This package provides a PyTorch implementation of low rank factorization machine models, the factorization model baselines and the common datasets in CTR prediction.
The code is running on the following datasets: Avazu, Criteo, and Movielens.

## The instructions contain three main steps: 
   - Step 1: Data Preprocessing and 
   - Step 2: Running the models on the preprocessed data.
   - Step 3: Analyzing the results

Moreover, this repository also contains a notebook `notebooks/inference_timing.ipynb` that demonstrates inference speedups
attained by low-rank models over pruned models, for different ad auction sizes and 
amounts of context fields.

## Available Datasets

* [MovieLens Dataset](https://www.kaggle.com/datasets/sherinclaudia/movielens)
* [Criteo Display Advertising Challenge](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset)
* [Avazu Click-Through Rate Prediction](https://www.kaggle.com/datasets/atirpetkar/avazu-ctr)

## Data Processing

Example of preprocessed dataset that will be obtained after running the preprocessing scripts on the initial datasets:

label, user_id, item_id, C1, C2, …  
&lt;label&gt;,10,2,3,17,11,15  
&lt;label&gt;,11,2,4,16,12,14


## How to preprocess Avazu dataset


1. Download the initial file (train, 6.31G) should be called data_avazu.csv
   https://www.kaggle.com/datasets/atirpetkar/avazu-ctr

2. put it to pytorch-tensorfm/torchfm/test-datasets/avazu/

3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.avazu.avazu_parsing import process_data 
    process_data()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-tensorfm/torchfm/test-datasets/avazu/

5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.

## How to preprocess Criteo dataset

1. Download the initial file (train.txt, 11.15G) should be called data_criteo.csv
   https://www.kaggle.com/datasets/mrkmakr/criteo-dataset
2. put it to pytorch-tensorfm/torchfm/test-datasets/criteo/
3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.criteo.criteo_parsing import CriteoParsing 
    CriteoParsing.do_preprocessing()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-tensorfm/torchfm/test-datasets/criteo/
5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.

## How to preprocess MovieLens dataset
1. Download the initial file users.dat, movies.dat, ratings.dat from
   https://www.kaggle.com/datasets/sherinclaudia/movielens
2. put it to pytorch-tensorfm/torchfm/test-datasets/movielens/
3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.movielens.movielens_parsing import MovielensParsing 
    MovielensParsing.process_data()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-tensorfm/torchfm/test-datasets/movielens/
5. Check you have enough (5G) available space and 
   proceed to run the ML models on the train-validation-test datasets.

## How to preprocess COMPAS dataset
1. Download the initial files from
   https://www.kaggle.com/datasets/danofer/compass
2. put it to pytorch-tensorfm/torchfm/test-datasets/COMPAS/ (The repo already includes the required filed)
3. To create train-validation-test datatsets, from the python  shell run: 

    from torchfm.torch_utils.parsing_datasets.COMPAS.Process_COMPAS import main
    main()

4. Check that now train.csv, test.csv, validation.csv and stored under /pytorch-tensorfm/torchfm/test-datasets/COMPAS/

# Instructions: How to run a dataset on a local environment

1. Copy the code to the dedicated folder. Install all requirements. 

2. In the shell redefine PYTHONPATH to point to your project root, .e.g,
export PYTHONPATH=$PYTHONPATH:/home/${USER}/pytorch-tensorfm/src:/home/${USER}/pytorch-tensorfm/src/main_functions

3. Edit pytorch-tensorfm/torchfm/torch_utils/constants.py file 
to have 
      - base_path_project pointing to your project root
      - Edit dataset_name to have the dataset you run on (avazu, criteo, movielens)
        E.g., dataset_name = movielens

    Edit pytorch-tensorfm/src/main_functions/run_processes.py file in examples folder, 
    to refer to the list of options to run:  currently, as example, it contains lst_tmp
    in "for tpl in lst_tmp:".
 
4. Copy train.csv/validation.csv/test.csv splitted datasets to be under pytorch-tensorfm/data/test-datasets/<dataset>/
(dataset is either criteo or avazu or movielens)
Then, open a python shell by running just: "python" command from the shell.

5. Check that you have enough space (at least 5G available) after all these steps in your local environment, e.g., by:
df -h /home/default/your_location
Otherwise remove non-required data (e.g., datasets you don’t use for the current run)

6. Check that you have pytorch-tensorfm/data/tmp_save_dir (if not create this folder)

7. If you are rerunning, check you don’t have a previous run. results stored (especially .log files - locking the next run), otherwise remove:
rm pytorch-tensorfm/data/tmp_save_dir/*

8. Run by 
python ./pytorch-tensorfm/src/main_functions/run_processes.py

9. After the run is done, the results are saved in 
/pytorch-tensorfm/data/tmp_save_dir/optuna_results.txt. 
Also, debug info is saved in /pytorch-tensorfm/data/tmp_save_dir/debug_info.txt

# Instructions: How to analyze the results
Open the notebook in `notebooks/analysis.ipynb`, modify the line
```python
files = glob.glob('../data/optuna_results*.txt')
```
to point to the appropriate paths with the result files saved in the previous stage, and run the notebook. It produces 
a Pandas table with the metrics and corresponding lifts of each dataset, and a
LaTeX code snippet you can put in a paper to share the results.
