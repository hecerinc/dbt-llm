'''
This class will be a base class for running steps in the validation pipeline.
The main method that must be implemented is the `run_evaluation` method which takes:
	- input: conversation
	- output: a dict object (easier to serialize), with the evaluation result
'''
from abc import ABC, abstractmethod
from typing import Dict
from openai import AzureOpenAI
from azureml.core import Workspace, Dataset
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('adherence_evaluation')
logger.setLevel(logging.INFO)

# Connect to OpenAI model
client = AzureOpenAI(
  api_key = "12858f5908ee4b17b986977b35d604dd",  
  api_version = "2023-05-15",
  azure_endpoint = "https://dbt-openai-usea2-assistants.openai.azure.com/"
)

class Evaluation:
    @abstractmethod
    def run_evaluation(self, conversation: str) -> Dict:
        raise NotImplementedError()

class AdherenceEvaluation(Evaluation):

    def __init__(self, number_iterations):
        #self.dataset = dataset
        self.number_iterations = number_iterations

        # Get evaluation dataset
        subscription_id = '8048e16e-5368-4d28-8d68-657559f557e7'
        resource_group = 'dbt-rg-openai'
        workspace_name = 'berkeley_dbt'

        workspace = Workspace(subscription_id, resource_group, workspace_name)

        dataset = Dataset.get_by_name(workspace, name='EvaluationPromptsChecklistComplete')
        self.checklist_df = dataset.to_pandas_dataframe()

        # standard = dataset['Standard']
        # standard_prompts = dataset['System Prompt']



    def evaluate_conversation(self, conversation):
        # Get proportion of adherence 
        # returns a dictionary with keys = standard name, value = proportion

        adherence_checklist = {}

        model = 'gpt4-1106'
        
        for index, row in self.checklist_df.iterrows():
            # call each prompt on each conversation
            messages=[
                     {"role": "system", "content": row['System Prompt']},
                     {"role": "user", "content": conversation}
                     ]
            adherence_checklist[row['Standard']] = int(get_openai_response_content(client, messages, model))
           
        
        return adherence_checklist
    

    def run_evaluation(self, conversation):

        evaluation_list = []
        for i in range(self.number_iterations):
            # Logger
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
