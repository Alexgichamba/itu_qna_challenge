# data_prep.py
# This script prepares the data for training and evaluation.

import os
import numpy as np
import json
import random

from prepare_docs import make_abbreviations, find_full_forms

# arrange into functions
def make_question_files():
    with open('data/TeleQnA_training.txt', 'r') as f_train_in:
        all_qns = json.load(f_train_in)
        # change to list
        all_qns_vals = list(all_qns.values())
        all_qns_keys = list(all_qns.keys())

    assert len(all_qns) == 1461, "The total number of questions is not 1461."

    # randomly select 61 questions for development
    dev_indices = random.sample(range(1461), 61)
    dev_qns = [all_qns_vals[i] for i in dev_indices]
    dev_keys = [all_qns_keys[i] for i in dev_indices]

    # use the remaining 1400 questions for training
    train_qns = [all_qns_vals[i] for i in range(1461) if i not in dev_indices]
    train_keys = [all_qns_keys[i] for i in range(1461) if i not in dev_indices]

    # save the development questions
    with open('data/qs_dev.txt', 'w') as f_dev_out:
        dev_qns = {dev_keys[i]: dev_qns[i] for i in range(len(dev_qns))}
        json.dump(dev_qns, f_dev_out, indent=4)

    # save the training questions
    with open('data/qs_train.txt', 'w') as f_train_out:
        train_qns = {train_keys[i]: train_qns[i] for i in range(len(train_qns))}
        json.dump(train_qns, f_train_out, indent=4)
    
    # merge the evaluation questions
    with open('data/qs_eval.txt', 'w') as f_eval_out:
        with open('data/TeleQnA_testing1.txt', 'r') as f_eval1_in:
            with open ('data/questions_new.txt', 'r') as f_eval2_in:
                eval1 = json.load(f_eval1_in)
                eval2 = json.load(f_eval2_in)
                eval_qns = {**eval1, **eval2}
                json.dump(eval_qns, f_eval_out, indent=4)

    print(f'Number of development questions: {len(dev_qns)}')
    print(f'Number of training questions: {len(train_qns)}')
    print(f'Number of evaluation questions: {len(eval_qns)}')

if __name__ == '__main__':

    random_seed = 20
    random.seed(random_seed)

    # make_question_files()

    # # make abbreviations
    # abbreviations = make_abbreviations(directory='data/rel18',
    #                                    abbrevs_heading='Abbreviations',
    #                                    abbrevs_heading_level=2)
    # all_abbreviations = dict(sorted(abbreviations.items()))
    # with open('data/abbreviations.txt', 'w') as f:
    #     for abbreviation, full_form in all_abbreviations.items():
    #         f.write(f"{abbreviation}: {full_form}\n")

    # replace abbreviations with full forms
    # find_full_forms()

    # generate a triplet dataset for training embeddings
    with open('data/qs_train.txt', 'r') as f_train_in:
        train_qns = json.load(f_train_in)
    
    # generate triplets: anchor, positive, negative
    # anchor: explanation
    # positive: correct answer
    # negative: incorrect answers
    # it should be like: {'set': {'explanation': '', 'correct': [''], 'incorrect': ['', '', ...]}}	
    triplets = {}
    for q_id, q_data in train_qns.items():
        options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]
        answer = q_data['answer']
        # answer is of the form 'option 1: ...'
        # need to find the option + number and build list of incorrect
        answer_option = answer.split(':')[0].strip()
        incorrect = [v for k, v in options if k.strip() != answer_option]
        correct = [v for k, v in options if k.strip() == answer_option]
        # add the triplet
        triplets[q_id] = {'explanation': q_data['explanation'],
                          'correct': correct,
                          'incorrect': incorrect}
    # save the triplets
    with open('data/triplets_train.txt', 'w') as f_triplets_out:
        json.dump(triplets, f_triplets_out, indent=4)
            
    # DO THE same for dev data
    with open('data/qs_dev.txt', 'r') as f_dev_in:
        dev_qns = json.load(f_dev_in)

    # generate triplets: anchor, positive, negative
    triplets = {}
    for q_id, q_data in dev_qns.items():
        options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]
        answer = q_data['answer']
        # answer is of the form 'option 1: ...'
        # need to find the option + number and build list of incorrect
        answer_option = answer.split(':')[0].strip()
        incorrect = [v for k, v in options if k.strip() != answer_option]
        correct = [v for k, v in options if k.strip() == answer_option]
        # add the triplet
        triplets[q_id] = {'explanation': q_data['explanation'],
                          'correct': correct,
                          'incorrect': incorrect}
    # save the triplets
    with open('data/triplets_dev.txt', 'w') as f_triplets_out:
        json.dump(triplets, f_triplets_out, indent=4)
        
        


