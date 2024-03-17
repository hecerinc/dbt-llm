import os, sys
import logging
from dotenv import load_dotenv
from datetime import datetime
import time

from openai import AzureOpenAI
from .sim_thread import SimThread

load_dotenv('.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'DEVELOPMENT')

pipeline_run_time_utc = datetime.utcnow()
run_id = pipeline_run_time_utc.strftime('%Y%m%dT%H%M%S')

assert OPENAI_API_KEY is not None, 'OpenAI API Key must be present in env variables'

# Connect to OpenAI model
openai_client = AzureOpenAI(
    api_key = OPENAI_API_KEY,
    api_version = "2023-05-15",
    azure_endpoint = "https://dbt-openai-usea2-assistants.openai.azure.com/"
)
subscription_id = '8048e16e-5368-4d28-8d68-657559f557e7'
resource_group = 'dbt-rg-openai'
workspace_name = 'berkeley_dbt'


with open(os.path.join('../prompts', 'dbt_system.prompt'), 'r', encoding='utf-8') as f:
    dbt_system_prompt = f.read()

with open(os.path.join('../prompts', 'user_impersonation.prompt'), 'r', encoding='utf-8') as f:
    persona_prompt = f.read()




def simulate_conversation(initial_prompt: str):
    st = SimThread(openai_client, dbt_system_prompt, None, persona_prompt, initial_prompt, 'dbt', 10, 10000)
    st.run_thread(verbose=True)
    # st.extend_thread(4, verbose=True)
    return st


def main():
    with open('../data/unique_prompts.txt', 'r', encoding='utf-8') as f:
        seed_prompts = f.readlines()
    if not seed_prompts:
        print('ERROR: no seed prompts found. Exiting.', file=sys.stderr)
        sys.exit(1)

    for initial_prompt in seed_prompts:
        initial_prompt = initial_prompt.strip()
        st = simulate_conversation(initial_prompt)
        st_eval_txt = st.msgs_to_eval() # this is what then gets passed to the evaluation prompt
        with open(f'{run_id}_result_console.txt', 'w', encoding='utf-8') as f:
            print(st.msgs_to_console(), file=f)
        with open(f'{run_id}_result_eval.txt', 'w', encoding='utf-8') as f:
            print(st_eval_txt, file=f)
# print(st.msgs_to_console())
# print(st_eval_txt)

