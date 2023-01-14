import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

import torch.nn as nn
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, DataCollatorWithPadding
from transformers import BertTokenizerFast, LongformerTokenizerFast
from transformers import BertForQuestionAnswering, LongformerForQuestionAnswering

import wandb
wandb.init(project="quac")

def add_end_pos(answers, docs):
    for answer, context in zip(answers, docs):
      end_pos = answer['answer_start'] + len(answer['text'])
      if context[answer['answer_start']:end_pos] == answer['text']:
        answer['answer_end'] = end_pos
      else:
        Exception('error..')
    return answers

def extract_info(data):
    questions, docs, answers, answer_candidates = [], [], [], []
    ids, is_impossible, yesno, followups = [], [], [], []

    for dialog in data:
        assert len(dialog)==1
        assert len(dialog['paragraphs'])==1
        assert len(dialog['paragraphs'][0]['qas'])==1

        # document/context
        doc = dialog['paragraphs'][0]['context']
        docs.append(doc)

        #others
        qas = dialog['paragraphs'][0]['qas'][0]
        ids.append(qas['id'])
        questions.append(qas['question'])
        answers.append(qas['answers'][0])
        is_impossible.append(qas['is_impossible'])
        yesno.append(qas['yesno'])
        followups.append(qas['followup'])
        answer_candidates.append(qas['answer_candidates'])

    answers = add_end_pos(answers, docs)
    return {'questions':questions, 'docs':docs, 'answers':answers, 'ids':ids, 'is_impossible':is_impossible, 'yesno':yesno, 'followups':followups, 'answer_candidates':answer_candidates}


def load_process_data(train_dir, val_dir):
    train = pd.read_json(train_dir)['data']
    val = pd.read_json(val_dir)['data']

    train = extract_info(train)
    val = extract_info(val)

    train = Dataset.from_dict(train)
    val = Dataset.from_dict(val)

    dataset = DatasetDict({'train': train, 'validation': val})

    return dataset

def add_token_positions(val, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # val.char_to_token(i, answers[i]['answer_start']) is the start_pos and it can be none
        start_positions.append(val.char_to_token(i, answers[i]['answer_start'], sequence_index=1))

        #this should not exist
        if(answers[i]['answer_end']==0):
            Exception('error...')
            #end_positions.append(val.char_to_token(i, answers[i]['answer_end'])) 
        else:
            end_positions.append(val.char_to_token(i, answers[i]['answer_end'] - 1, sequence_index=1))

         # if None, the answer passage has been truncated
         # Here is not a good approach
        if start_positions[-1] is None:
            #print('start_positions[-1] is None')
            start_positions[-1] = tokenizer.model_max_length

        if end_positions[-1] is None:
            #print('end_positions[-1] is None')
            end_positions[-1] = tokenizer.model_max_length

    return start_positions, end_positions

def prepare_model(trainSet,valSet,tokenizer):

    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', cache_dir="bert_base/")
    #model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096', cache_dir="longformer/")
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
        run_name="bert-large-uncased-finetuned-quac",
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

    train_dir = 'data/quac_train.json' 
    val_dir = 'data/quac_dev.json' 
    dataset = load_process_data(train_dir, val_dir)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir="bert_base/")
    #tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', cache_dir="longformer/")

    def encode(examples):
      """Mapping function to tokenize the sentences passed with truncation"""
      encoding = tokenizer(examples["questions"], examples["docs"], truncation=True, padding="max_length",
                       max_length=512, return_special_tokens_mask=True)

      start_positions, end_positions = add_token_positions(encoding, examples["answers"], tokenizer)
      encoding.update({'start_positions': start_positions, 'end_positions': end_positions})
      return encoding

    train =  dataset["train"].map(encode, batched=True)
    val =  dataset["validation"].map(encode, batched=True)

    trainer_quac = prepare_model(train,val,tokenizer)

    trainer_quac.train()

    return


if __name__ == "__main__":
  main()