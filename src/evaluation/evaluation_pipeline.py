'''
Takes an input file with the conversations to be evaluated and runs them through one or several evaluations.
Outputs results to blob storage. TODO: figure out if this is the best place.
'''

import os, sys
import json
import argparse
import logging
from typing import List, Dict
from datetime import datetime

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))

from config.config import Config as config
from evaluation.evaluation import Evaluation
from utils.blob_storage import read_blob, write_blob

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('evaluation_pipeline')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
args = parser.parse_args()


DBT_CONTAINER = config.DBT_CONTAINER
EVAL_BLOB_PATH = config.EVAL_BLOB_PATH
BLOB_CONNECTION_STRING = os.getenv('BLOB_CONNECTION_STRING', None)

assert BLOB_CONNECTION_STRING is not None, "BLOB_CONNECTION_STRING is required to run evaluation pipeline."

pipeline_run_time_utc = datetime.utcnow()
run_id = pipeline_run_time_utc.strftime('%Y%m%dT%H%M%S')


def get_evaluation_steps() -> List[Evaluation]:
    '''
    This is where we would put the different evaluation steps
    '''
    return []


def main():
    conversations = read_blob(connection_string=BLOB_CONNECTION_STRING, container=DBT_CONTAINER, path=args.input_path).splitlines()
    conversations = [json.loads(c) for c in conversations]


    evaluation_steps = get_evaluation_steps()

    evaluation_results: List[Dict] = []

    for convo in conversations:
        for evstep in evaluation_steps:
            result = evstep.run_evaluation(convo)
            evaluation_results.append(result)

    for eval_result in evaluation_results:
        result = '\n'.join(json.dumps(eval_result))
        write_blob(data=result, connection_string=BLOB_CONNECTION_STRING, container=DBT_CONTAINER, path=f'{EVAL_BLOB_PATH}/{run_id}_evaluation_result.jsonl', overwrite=True)


if __name__ == '__main__':
    main()
