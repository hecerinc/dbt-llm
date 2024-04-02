'''
Takes an input file with the conversations to be evaluated and runs them through one or several evaluations.
Outputs results to blob storage. TODO: figure out if this is the best place.
'''

import os, sys
import json
import argparse
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime


sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))

from config.config import Config as config
from evaluation.evaluation import Evaluation
from evaluation.adherence_evaluation import AdherenceEvaluation
from evaluation.skill_presence_evaluation import SkillPresenceEvaluation
# from utils.blob_storage import read_blob, write_blob

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('evaluation_pipeline')
logger.setLevel(logging.INFO)


DBT_CONTAINER = config.DBT_CONTAINER
EVAL_BLOB_PATH = config.EVAL_BLOB_PATH
ADHERENCE_EVAL_ITERATIONS = config.ADHERENCE_EVAL_ITERATIONS
# BLOB_CONNECTION_STRING = os.getenv('BLOB_CONNECTION_STRING', None)

# assert BLOB_CONNECTION_STRING is not None, "BLOB_CONNECTION_STRING is required to run evaluation pipeline."

pipeline_run_time_utc = datetime.utcnow()
run_id = pipeline_run_time_utc.strftime('%Y%m%dT%H%M%S')

VALID_EVAL_NAMES = ['all', 'adherence', 'skill_presence']


def get_evaluation_steps(eval_name: str = 'all') -> List[Evaluation]:
    '''
    This is where we would put the different evaluation steps
    '''
    if eval_name.lower() == 'all':
        return [AdherenceEvaluation(ADHERENCE_EVAL_ITERATIONS), SkillPresenceEvaluation()]
    elif eval_name.lower() == 'adherence':
        return [AdherenceEvaluation(ADHERENCE_EVAL_ITERATIONS)]
    elif eval_name.lower() == 'skill_presence':
        return [SkillPresenceEvaluation()]
    else:
        raise Exception(f'Eval name \'{eval_name}\' is invalid. Valid eval names are: {VALID_EVAL_NAMES}')



def run_evaluations(input_path: str, output_path: str, eval_name: str = 'all'):
    '''
    The input_path is a CSV file with a `conversation_id` and `conversation` column.
    '''
    conversation_df = pd.read_csv(input_path)

    evaluation_steps = get_evaluation_steps(eval_name)

    evaluation_results: List[Dict] = []

    n_convos = len(conversation_df)

    for _i, row in conversation_df.iterrows():
        logger.info(f'Running conversation: ({int(_i)+1}/{n_convos})')
        convo = str(row['conversation'])
        convo_id = row['conversation_id']
        for evstep in evaluation_steps:
            result = evstep.run_evaluation(convo)
            evaluation_results.append({
                'conversation_id': convo_id,
                'evaluation_name': evstep.name,
                'result': result
                })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f)
    # for eval_result in evaluation_results:
    #     result = '\n'.join(json.dumps(eval_result))
    #     write_blob(data=result, connection_string=BLOB_CONNECTION_STRING, container=DBT_CONTAINER, path=f'{EVAL_BLOB_PATH}/{run_id}_evaluation_result.jsonl', overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--eval_name")
    args = parser.parse_args()
    run_evaluations(args.input_path, args.output_path, args.eval_name)


if __name__ == '__main__':
    main()
