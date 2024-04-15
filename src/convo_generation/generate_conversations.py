import os, sys
import pickle
import logging
from dotenv import load_dotenv
from datetime import datetime
import time
import argparse
import pandas as pd

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))
from openai import AzureOpenAI
from convo_generation.sim_thread import SimThread

load_dotenv('.env')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('generate_conversation')
logger.setLevel(logging.INFO)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'DEVELOPMENT')
DEBUG = os.getenv('DEBUG', 'false')

DEBUG = True if DEBUG.lower() == 'true' else False


parser = argparse.ArgumentParser()

# For testing, limit the number of conversations generated
parser.add_argument('-l', '--limit', required=False)
# Limit the number of messages per conversation
parser.add_argument('-n', '--n_msgs', required=False, default=10)
parser.add_argument('-o', '--output', required=False, default='result.pickle')

args = parser.parse_args()

convo_limit = int(args.limit) if args.limit else None
convo_n_msgs = int(args.n_msgs) if args.n_msgs else None
outfile = os.path.join('output', args.output)


pipeline_run_time_utc = datetime.utcnow()
run_id = pipeline_run_time_utc.strftime('%Y%m%dT%H%M%S')

assert OPENAI_API_KEY is not None, 'OpenAI API Key must be present in env variables'
oai_endpoint = os.getenv('OAI_ENDPOINT')

# Connect to OpenAI model
openai_client = AzureOpenAI(
    api_key = OPENAI_API_KEY,
    api_version = "2023-05-15",
    azure_endpoint = f"https://{oai_endpoint}.openai.azure.com/"
)

with open(os.path.join('..', 'prompts', 'dbt_system.prompt'), 'r', encoding='utf-8') as f:
    dbt_system_prompt = f.read()

with open(os.path.join('..', 'prompts', 'user_impersonation.prompt'), 'r', encoding='utf-8') as f:
    persona_prompt = f.read()



def simulate_conversation(initial_prompt: str):
    st = SimThread(openai_client, dbt_system_prompt, None, persona_prompt, initial_prompt, 'dbt', convo_n_msgs, 10000)
    st.run_thread(verbose=DEBUG)
    # st.extend_thread(4, verbose=True)
    return st


def main():
    # Get the input prompts
    logger.info('Reading unique prompts')

    seed_prompts = pd.read_csv('../data/prompts_with_ids.tsv', delimiter='\t')
    convo_ids = list(seed_prompts['id'])
    seed_prompts = list(seed_prompts['Initial Message'])

    if convo_limit:
        seed_prompts = seed_prompts[:convo_limit]

    if not seed_prompts:
        logger.exception('No seed prompts found. Exiting.')
        sys.exit(1)

    logger.info(f'Read {len(seed_prompts)} unique prompts.')

    results = {}

    for i, initial_prompt in enumerate(seed_prompts):
        try:

            initial_prompt = initial_prompt.strip()
            convo_id = convo_ids[i]
            logger.info(f'Running prompt {convo_id}')

            t_start = time.perf_counter()
            st = simulate_conversation(initial_prompt)
            t_stop = time.perf_counter()

            elapsed_time = t_stop - t_start

            dbt = st.get_agent_info('dbt')
            persona = st.get_agent_info('persona')

            results[convo_id] = {
                'id': convo_id,
                'result': {
                    'dbt': dbt,
                    'persona': persona,
                    'messages': st.thread_msgs,
                    'eval_messages': st.msgs_to_eval()
                },
                'prompt': initial_prompt,
                'thread_stats': st.thread_stats,
                'elapsed_time': elapsed_time
            }

            logger.info(f'Elapsed time: {elapsed_time}')
        except Exception as e:
            logger.error(f'Failed to process prompt {i+1}')
            logger.exception(e)

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
    logger.info('Done')


if __name__ == '__main__':
    main()
