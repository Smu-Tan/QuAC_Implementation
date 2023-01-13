import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import os
import json
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator, EarlyStoppingCallback, IntervalStrategy, DataCollatorWithPadding
import math
from transformers import BertTokenizerFast,DistilBertTokenizerFast
from transformers import BertForQuestionAnswering,DistilBertForQuestionAnswering

import wandb
wandb.init(project="quac")


def jsonToLists(contents):
  texts=[]
  questions=[]
  answers=[]
  for data in contents:
    for txt in data['paragraphs']: # every text,
      text = txt['context']
      for qa in txt['qas']: # has many questions,
        question = qa['question']
        for answer in qa['answers']: # questions have answers.
          texts.append(text)
          questions.append(question)
          answers.append(answer)            
                  
  return texts,questions,answers

def add_end_idx(answers, contexts):
    # get the character position at which every answer ends and store it
    for answer, context in zip(answers, contexts):
      end_idx = answer['answer_start'] + len(answer['text'])

      if context[answer['answer_start']:end_idx] == answer['text']:
        answer['answer_end'] = end_idx
      # Sometimes SQuAD answers are off by one or two characters, so ...
      elif context[answer['answer_start']-1:end_idx-1] == answer['text']:
        answer['answer_start'] = answer['answer_start'] - 1
        answer['answer_end'] = end_idx - 1
      elif context[answer['answer_start']-2:end_idx-2] == answer['text']:
        answer['answer_start'] = answer['answer_start'] - 2
        answer['answer_end'] = end_idx - 2
    return answers, contexts

def add_token_positions(encodings, answers, tokenizer):
    # convert character start/end positions to token start/end positions.
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        if(answers[i]['answer_end']==0):
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'])) 
        else:
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

         # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
                   
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    return encodings,answers, tokenizer

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
      self.encodings = encodings

    def __getitem__(self, idx):
      return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
      return len(self.encodings.input_ids)

def load_data():
    jsonTrain = "./quac_train.json"
    jsonVal = "./quac_dev.json"
    
    file_data = pd.read_json(jsonTrain)
    json_data = file_data['data']

    size = math.floor(json_data.shape[0]*0.7)
    trainContents = file_data['data'].loc[0:size]

    file_data = pd.read_json(jsonVal)
    json_data = file_data['data']

    size = math.floor(json_data.shape[0])
    valContents = file_data['data'].loc[0:size]
    print('data loaded!')

    trainTexts,trainQuestions,trainAnswers=jsonToLists(trainContents)
    valTexts,valQuestions,valAnswers=jsonToLists(valContents)

    trainAnswers, trainTexts = add_end_idx(trainAnswers, trainTexts)
    valAnswers, valTexts = add_end_idx(valAnswers, valTexts)
    print('processed files...')

    return trainTexts,trainQuestions,trainAnswers, valTexts,valQuestions,valAnswers


def process_data(trainTexts,trainQuestions,trainAnswers, valTexts,valQuestions,valAnswers):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    trainEncodings=tokenizer(trainTexts,trainQuestions, truncation=True, padding=True, max_length=512)
    valEncodings=tokenizer(valTexts,valQuestions, truncation=True, padding=True, max_length=512)

    # apply function to our data
    trainEncodings, trainAnswers, tokenizer = add_token_positions(trainEncodings, trainAnswers, tokenizer)
    valEncodings, valAnswers, tokenizer = add_token_positions(valEncodings, valAnswers, tokenizer)

    # create the corresponding datasets
    trainSet = SquadDataset(trainEncodings)
    valSet = SquadDataset(valEncodings)

    print('processed datasets...')
    return trainSet,valSet,tokenizer

def prepare_model(trainSet,valSet,tokenizer):

    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
    model.cuda()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print('preparing model...')

    args = TrainingArguments(
        "bert-base-cased-finetuned-quac",
        evaluation_strategy = "steps",
        eval_steps=250,
        save_strategy = "steps",
        learning_rate=2e-5,
        adafactor=True,
        per_device_train_batch_size=32,
        #gradient_accumulation_steps=4,
        per_device_eval_batch_size=20,
        logging_steps = 50,
        num_train_epochs=50,
        group_by_length=True,
        weight_decay=0.01,
        fp16=True,
        #warmup_ratio=0.02,
        save_total_limit = 3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="bert-base-cased-finetuned-quac",
      )

    trainer_quac = Trainer(
        model=model,
        args=args,
        train_dataset=trainSet,
        eval_dataset=valSet,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
      )

    print('ready to train!')
    return trainer_quac


def main():

    trainTexts,trainQuestions,trainAnswers, valTexts,valQuestions,valAnswers = load_data()
    
    trainSet,valSet,tokenizer = process_data(trainTexts,trainQuestions,trainAnswers, valTexts,valQuestions,valAnswers)

    trainer_quac = prepare_model(trainSet,valSet,tokenizer)

    trainer_quac.train()

    return


if __name__ == "__main__":
  main()