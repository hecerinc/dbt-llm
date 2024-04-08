# BerkeleyDBT

## About The Project

BerkeleyDBT is an AI chatbot that helps users navigate their emotions through Dialectical Behavior Therapy (DBT). Our project offers a supportive and interactive platform where users can engage in meaningful conversations, receive personalized feedback, and practice DBT techniques to help better manage their emotional challenges. 

## High Level Implementation

For implementation, we hosted in AzureOpenAI and built on top of gpt-4. We created custom system prompts to specifically train our chatbot on DBT related concepts. 

## Overview of Files

### convo_generation folder

- generate_conversations.py

  This Python script utilizes the OpenAI API through an AzureOpenAI client to simulate conversations based on seed prompts. These conversations are artificially created by creating varying 'personas' that engage with our chatbot. How this works is it first reads seed prompts that we manually created and imported into a CSV file. Next, it simulates conversations through functions created in our SimThread class, found in sim_thread.py. And lastly, it saves the conversation generated to a pickle file, specified by the user. The script also includes logging and error handling functionalities to track execution.

- sim_thread.py

  This is the SimThread class, which is designed to simulate the conversations described in the generate_conversations.py file. The class manages conversation threads between two agents, 'BerkeleyDBT' and the varying 'personas' we created from seed prompts. BerkeleyDBT and these personas will alternate responses with each other until a conversation is produced to a desired length. The class includes methods for generating responses, updating conversation statistics, and printing messages to the console or evaluation format.

### data folder

- Evaluation_prompts.tsv
  
  This tab-separated values file lists 23 DBT standards found in the DBT Adherence and Fidelity Checklist with its corresponding system prompt, ID, and description category, 

- dbt_skills_ref.csv

  This tab-separated values file lists each DBT skill from the DBT skills manual with its corresponding module focus, module name, module category, and skill description.  

- prompts_with_ids.tsv

  This tab-separated values file lists each persona prompt with the respective DBT skill that should be applied as well as its corresponding ID.

- skill_presence_base_prompts.txt

  The text file instructs a reviewer to rate whether a therapist discussed a specified DBT skill during a session based on specific criteria, providing a rating of "1" if the skill was discussed and "0" if it was not.

- skill_presence_regex.json

  The json file is a collection of RegEx expression to help identify whether`` or not a specified DBT skill was mentioned in a conversation prompt.

### evaluation folder

- adherence_evaluation.py

  This python script include the AdherenceEvaluation class, which extends the Evaluation class. It is primarily designed to evaluate the adherence of a conversation to predefined standards using the OpenAI API. It first reads the 23 DBT standards found in Evaluations_prompts.tsv. Next, it iteratively evaluates a given conversation against these 23 standards using the OpenAI chat API. Lastly, it calculates the proportion of adherence for each criterion across multiple iterations, returning the aggregated results. 

- evaluation.py

  This python script includes the Evaluation class, which serves as an abstract base class for running the steps in the validation pipeline. 

- evaluation_pipeline.py

  This python script takes an input file with the conversations to be evaluated and runs them through one or several evaluations.

- skill_presence_evaluation.py

  This python script evaluates a conversation on whether or not a specific DBT skill was present. It returns a list of skill IDs that are present within each of the conversations.

### Complete_pipeline.ipynb

  This python notebook provides a complete depiction of our pipeline process. Here, we generate conversations, evaluate our output through adherence and skills presence, and display our results. 

Copy the `env.sample` to `.env` and add the OPENAI_API_KEY.

## Contact Information

Hector Rincon - hrincon@berkeley.edu

Robert Mueller - robertmueller@berkeley.edu

Andrew Loeber - aloeber@berkeley.edu

John Van - johnvan@berkeley.edu

## Acknowledgements
