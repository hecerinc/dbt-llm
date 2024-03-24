import os, sys
import pickle
import logging
from dotenv import load_dotenv
from datetime import datetime
import time
import argparse

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
parser.add_argument('-f', '--prompt-file', required=True)


args = parser.parse_args()

uprompt_file = args.prompt_file

pipeline_run_time_utc = datetime.utcnow()
run_id = pipeline_run_time_utc.strftime('%Y%m%dT%H%M%S')

CONVERSATION_MESSAGE_LIMIT = 10

assert OPENAI_API_KEY is not None, 'OpenAI API Key must be present in env variables'

oai_endpoint = 'dbt-openai-usea2-assistants'

# Connect to OpenAI model
openai_client = AzureOpenAI(
    api_key = OPENAI_API_KEY,
    api_version = "2023-05-15",
    azure_endpoint = f"https://{oai_endpoint}.openai.azure.com/"
)
subscription_id = '8048e16e-5368-4d28-8d68-657559f557e7'
resource_group = 'dbt-rg-openai'
workspace_name = 'berkeley_dbt'


with open(os.path.join('../prompts', 'dbt_system.prompt'), 'r', encoding='utf-8') as f:
    dbt_system_prompt = f.read()

with open(os.path.join('../prompts', 'user_impersonation.prompt'), 'r', encoding='utf-8') as f:
    persona_prompt = f.read()




def simulate_conversation(initial_prompt: str):
    st = SimThread(openai_client, dbt_system_prompt, None, persona_prompt, initial_prompt, 'dbt', CONVERSATION_MESSAGE_LIMIT, 10000)
    st.run_thread(verbose=DEBUG)
    # st.extend_thread(4, verbose=True)
    return st


def main():
    # Get the input prompts
    logger.info('Reading unique prompts')

    seed_prompts = pd.read_csv('../data/src/data/Prompt to Skill Pairs.tsv', delimiter='\t')
    seed_prompts = list(seed_prompts['Initial Message'])[:3] #FROM ROB: NEED TO CHANGE THIS TO CONTROL FOR NUMBER OF CONVERSATIONS WITH SOME CLASS

    if not seed_prompts:
        print('ERROR: no seed prompts found. Exiting.', file=sys.stderr)
        sys.exit(1)

    logger.info(f'Read {len(seed_prompts)} unique prompts.')

    results = {}

    for i, initial_prompt in enumerate(seed_prompts):
        try:

            initial_prompt = initial_prompt.strip()
            logger.info(f'Running prompt {i+1}')

            t_start = time.perf_counter()
            st = simulate_conversation(initial_prompt)
            t_stop = time.perf_counter()

            elapsed_time = t_stop - t_start

            dbt = st.get_agent_info('dbt')
            persona = st.get_agent_info('persona')

            results[i] = {
                'id': i+1,
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

        # st_eval_txt = st.msgs_to_eval() # this is what then gets passed to the evaluation prompt
        # with open(f'{run_id}_result_console.txt', 'w', encoding='utf-8') as f:
        #     print(st.msgs_to_console(), file=f)
        # with open(f'{run_id}_result_eval.txt', 'w', encoding='utf-8') as f:
        #     print(st_eval_txt, file=f)

    with open('result.pickle', 'wb') as f:
        pickle.dump(results, f)
# print(st.msgs_to_console())
# print(st_eval_txt)


if __name__ == '__main__':
    main()
