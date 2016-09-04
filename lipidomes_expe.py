# -*- coding: utf-8 -*-
from __future__ import print_function, division
__author__ = 'maoss2'
from future.utils import iteritems
import logging
import numpy as np
import pandas as pd
import random
from time import time
import h5py
import seaborn as sns
import os
from joblib import Parallel, delayed
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from graalpy.dataset_partition import TrainTestSplitter
from graalpy.experiment import local_test, job_dispatcher_submit
from graalpy.learners import classification_lasso, scm
from graalpy.results_dataframe import load_results_recursively
from graalpy.utils.pyjobber_utils import get_latest_experiment_folder
from graalpy.utils.results import compare_algorithms
from graalpy.utils.tableView import TestInfo, Resume
from graalpy.dataset_partition import get_train_and_test_datasets

CONTEXT = 'PC_lipidome'

def build_dataset(file_path, output_path, name, random_state_value):
    """ Build the dataset """
    dlb_data = pd.read_excel(file_path, sheetname="DLB")
    dlb_data = dlb_data.T
    control_data = pd.read_excel(file_path, sheetname="Neurological controls")
    control_data = control_data.T
    frames = [dlb_data, control_data.sample(n=51, random_state=random_state_value, axis=0)]
    data = pd.concat(frames)
    # 1 phenotype ill; -1 phenotype control
    labels = np.append(np.ones((51,), dtype=np.int), -1 * np.ones((51,), dtype=np.int))
    patients_name = data.index.values

    # the control patients name (to be written somewhere) in the hdf5?
    patients_name = patients_name.astype("string")

    # build array for graalpy
    data = data.as_matrix()
    x_zipped = zip(data, labels, patients_name)
    # shuffle the data
    random.shuffle(x_zipped)
    data, target, patients_name = zip(*x_zipped)
    # patients name and position(index) in the dataset
    name_and_position = [(el, i) for i, el in enumerate(patients_name)]

    f = h5py.File(os.path.join(output_path, '{}.h5'.format(name)), 'w')
    f.create_dataset('data', data=data)
    f.create_dataset('target', data=target)
    # Reminder: goes into fit_parameters in graalpy
    f.create_dataset('name_and_position', data=name_and_position)

def paralel_build_dataset(indices, r):
    """ A parallel caller ot the build_dataset function"""
    build_dataset(file_path="/home/maoss2/PycharmProjects/lipidomes/PC_DLBNeuroControlBennettv2.xlsx",
                  output_path="/home/maoss2/Documents/Doctorat/Datasets_repository/%s" % CONTEXT,
                  name="dataset_Number_%d" % indices,
                  random_state_value=r)

def run_dataset_builder():
    # build the dataset once: To be commented after
    start_time = time()
    Parallel(n_jobs=8)(delayed(paralel_build_dataset)(indices, r)
                       for indices, r in enumerate(random.sample(np.arange(1, 10000), 100)))
    end_time = time() - start_time
    print (end_time)

def run_dispatcher(algorithms):
    """ Dispatch example, on a context of datasets, choosing the best hyperparameters with cross-validation.
    This uses job dispatcher, which automatically adapts to the computing platform and saves the results in the
    EXPERIMENT_FOLDER.

    """
    for name, algo in iteritems(algorithms):
        job_dispatcher_submit(name=name,  # Name of the algorithm (used for naming the results directory)
                              context_name=CONTEXT,  # The name of the context on which to run the algorithm
                              learner=algo["learner"],  # An instantiated learner object
                              params_dict=algo["params_dict"],  # The hyperparameter dictionnary
                              n_folds=5,  # Number of cross-validation folds
                              n_jobs=8,  # Number of processes per node (can be specified using the JD_PPN environment variable)
                              n_nodes=1,  # Number of nodes (use 1 for your personal computer) 4 pour ls31
                              walltime=24 * 60 * 60,  # The walltime in seconds
                              splitter=TrainTestSplitter(train_ratio=0.7),  # Proportion of the data to use for training
                              )

def get_dataset_report(datset_path):
    """ Get the name and indices of the data in the train and test. Graalpy just takke the 1st train_ratio elements of
    the dataset, then it'll be easy to get them separatly by just taking the SAME train_ratio.
    Note: If somehow graalpy made a shuffle before in the train_test_split, this is fucked up!!! """
    from glob import glob
    os.chdir(datset_path)
    with open("Names_Of_patients", "ab") as report:
        for fich in glob("*.h5"):
            train_data, test_data = get_train_and_test_datasets(dataset_path=fich, split_ratio=0.7)
            report.write("Dataset is: %s\n" % fich)
            report.write("train infos: %s\n" % train_data.fit_parameters["name_and_position"][:71])
            report.write("test infos: %s\n" % test_data.fit_parameters["name_and_position"][72:])
            report.write("\n")

def benchmark_tables(algorithms, hp_on_test=False):
    """ Example: Load results for two algorithms, keep best cross-validation results,
        build a single dataframe (indexing with the dataset name), output a LaTeX table.

    """
    if hp_on_test:
        metric_to_optimize = 'test__zero_one_loss'
    else:
        metric_to_optimize = 'cv_mean__valid__zero_one_loss'

    results_by_algo = {}
    for name in algorithms:
        results_by_algo[name] = load_results_recursively(get_latest_experiment_folder(name, CONTEXT))
    print(compare_algorithms(results_by_algo.items(), metric_to_optimize=metric_to_optimize, convert_to_latex=True))
    print()

def benchmark_pdf_report(algorithms, hp_on_test=False):
    """
    Build a PDF comparing all the algorithms.

    """
    if hp_on_test:
        metric_to_optimize = 'test__zero_one_loss'
    else:
        metric_to_optimize = 'cv_mean__valid__zero_one_loss'
    minimize = True
    # The metrics to show in the tables. Set to None to keep the default values.
    # metrics_to_show = None

    # metrics on test
    metrics_to_show = ['test__zero_one_loss',
                       #'cv_mean__valid__zero_one_loss',
                       #'test__precision',
                       'test__sensitivity',
                       'test__specificity',
                       'test__accuracy',
                       ]
    # # because the pdf table wont take it at all, build the zero_one_loss_per_example in another file
    # metrics_to_show = ['test__zero_one_loss_per_example']

    # # metrics on train
    # metrics_to_show = ['train__zero_one_loss',
    #                    'train__zero_one_loss_per_example',
    #                    'cv_mean__valid__zero_one_loss',
    #                    'train__precision',
    #                    'train__sensitivity',
    #                    'train__specificity',
    #                    'train__accuracy',
    #                    ]

    # # because the pdf table wont take it at all, build the zero_one_loss_per_example in another file
    # metrics_to_show = ['train__zero_one_loss_per_example']

    metric_to_compare = 'test__zero_one_loss'  # The metric to compare in the comparison table.
    table_name = 'Test Risk Comparison'  # The name of the table.

    test_list = []
    for name in algorithms:
        latest_folder = get_latest_experiment_folder(name, CONTEXT)
        test_list.append(TestInfo(latest_folder, name.replace("_", "-"),
                                  metric_to_optimize=metric_to_optimize,
                                  minimize=minimize,
                                  tie_breaking_functions_ordered_dict=None,
                                  metrics_to_show=metrics_to_show))

    resume = Resume(test_list, sgnTest=True, individualInfo=True, metricToCompare=metric_to_compare, comparisonTableName=table_name)
    resume.texDoc.pdf()

def main():
    # ------------------------------
    #    Algorithm Specifications
    # ------------------------------
    algorithms = {
        "RBF_SVC": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('svm', SVC(kernel='rbf',random_state=42))]),
            "params_dict": {
                'svm__C': np.logspace(-3, 3, 15),
                'svm__gamma': np.logspace(-3, 3, 15)
            }
        },

        "LINEAR_SVC": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('svmLinear', SVC(kernel='linear', random_state=42))]),
            "params_dict": {
                'svmLinear__C': np.logspace(-3, 3, 15)
            }
        },

        "DecisionTrees": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('dT', DecisionTreeClassifier(random_state=42))]),
            "params_dict": {
                'dT__max_depth': [2, 4, 6, 8],
                'dT__min_samples_split': [2, 3, 4, 5],
                'dT__max_features': [10, 20, 30, 40, 50]
            }
        },

        "LinearSvc": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('lsvc', LinearSVC(penalty="l1", dual=False))]),
            "params_dict": {
                'lsvc__C': np.logspace(-3, 3, 15)
            }
        },

        "LASSO": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('lasso', classification_lasso.LassoClassifier(alpha=0.1))]),
            "params_dict": {
                'lasso__alpha': np.logspace(-3, 3, 15)
            }
        },

        "RandomForest": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('rf', RandomForestClassifier())]),
            "params_dict": {
                    'rf__n_estimators': [100, 200, 250, 300, 350, 400],
                    'rf__max_features': [10, 20, 30, 40, 50]
            }
        },

        "SCM": {
            "learner": Pipeline([('scaling', StandardScaler()),
                                 ('scm', scm.SCMClassifier())]),
            "params_dict": {
                'scm__p': np.logspace(-2, 2, 5),
                'scm__max_attributes': (1, 2, 5, 7, 10)
            }
        }

    }

    # ------------------------------
    #        LAUNCHING JOBS
    # ------------------------------
    # run_dispatcher(algorithms)

    # ------------------------------
    #        VIEWING RESULTS
    # ------------------------------

    # Benchmark tables in LaTeX
    # print("Test Risk Benchmark (HP selection: Cross-validation)")
    # benchmark_tables(algorithms, hp_on_test=False)
    # print()
    # print("Test Risk Benchmark (HP selection: Testing set)")
    # benchmark_tables(algorithms, hp_on_test=True)

    # Benchmark PDF report (including Poisson binomial test)
    benchmark_pdf_report(algorithms)

if __name__ == '__main__':
    #run_dataset_builder()
    # main()
    get_dataset_report("/home/maoss2/Documents/Doctorat/Datasets_repository/PC_lipidome")