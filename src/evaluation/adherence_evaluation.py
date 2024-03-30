import sys, os
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('adherence_evaluation')
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))

from .evaluation import Evaluation
from utils.openai_api import get_openai_response_content

load_dotenv('.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
OAI_ENDPOINT = os.getenv('OAI_ENDPOINT', None)
MODEL_DEPLOYMENT = os.getenv('MODEL_DEPLOYMENT', 'gpt4-1106')

dir_path = os.path.dirname(os.path.realpath(__file__))

class AdherenceEvaluation(Evaluation):

    name = 'Adherence'

    def __init__(self, number_iterations):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
        OAI_ENDPOINT = os.getenv('OAI_ENDPOINT', None)
        MODEL_DEPLOYMENT = os.getenv('MODEL_DEPLOYMENT', 'gpt4-1106')
        #self.dataset = dataset
        self.number_iterations = number_iterations
        eval_prompts_path = os.path.join(dir_path, '../data/Evaluation Prompts.tsv')
        self.checklist_df = pd.read_csv(eval_prompts_path, delimiter = '\t')
        self.openai_client = AzureOpenAI(
                api_key = OPENAI_API_KEY,
                api_version = "2023-05-15",
                azure_endpoint = f"https://{OAI_ENDPOINT}.openai.azure.com/"
                )


        # Get evaluation dataset
        # subscription_id = '8048e16e-5368-4d28-8d68-657559f557e7'
        # resource_group = 'dbt-rg-openai'
        # workspace_name = 'berkeley_dbt'

        # workspace = Workspace(subscription_id, resource_group, workspace_name)

        # dataset = Dataset.get_by_name(workspace, name='EvaluationPromptsChecklistComplete')
        # self.checklist_df = dataset.to_pandas_dataframe()

        # standard = dataset['Standard']
        # standard_prompts = dataset['System Prompt']



    def evaluate_conversation(self, conversation):
        # Get proportion of adherence
        # returns a dictionary with keys = standard name, value = proportion

        adherence_checklist = {}


        n_criterion = len(self.checklist_df)
        i = 0
        for _i, row in self.checklist_df.iterrows():
            # call each prompt on each conversation
            criterion_name = row['Standard']
            logger.info(f'Running criterion: {criterion_name} ({int(i)+1}/{n_criterion})')
            messages = [
                    {"role": "system", "content": row['System Prompt']},
                    {"role": "user", "content": conversation}
                    ]
            # result = get_openai_response_content(self.openai_client, messages, MODEL_DEPLOYMENT)

            response = self.openai_client.chat.completions.create(
                model=MODEL_DEPLOYMENT,
                messages=messages
            )
            msg_txt = response.choices[0].message.content
            if msg_txt is None:
                logger.error('Empty msg_txt', msg_txt)
            # print(msg_txt)
            adherence_checklist[criterion_name] = int(msg_txt) if msg_txt is not None else msg_txt
            i += 1


        return adherence_checklist


    def run_evaluation(self, conversation: str):
        evaluation_list = []
        for i in range(self.number_iterations):
            logger.info(f"Running iteration {i}")

            result = self.evaluate_conversation(conversation)
            evaluation_list.append(result)

        # Aggregate all iterations
        evaluation_keys = evaluation_list[0].keys()

        proportion_adherence = {}
        for criterion in evaluation_keys:
            sum_of_key = 0
            for evaluation in evaluation_list:
                if criterion in evaluation:
                    sum_of_key += evaluation[criterion]
            proportion_adherence[criterion] = (sum_of_key / self.number_iterations)
        return proportion_adherence
