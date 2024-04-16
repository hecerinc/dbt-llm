# Berkeley DBT

## About The Project

Berkeley DBT is an open source development pipeline intended to provide a simplified process for mental health researchers to rapidly iterate, test, and deploy GPT-based virtual assistants that are deeply integrated with Dialectical Behavior Therapy (DBT) principles. Our focus was on building a framework for generating synthetic chat conversations from a corpus of patient profiles, and conducting automated performance evaluations measuring a chatbot's adherence to DBT care quality standards, as well as the quality of its DBT skill recommendations.

For our own implementation, we deployed an instance of GPT-4 Turbo via Azure AI Studio.

## High Level Implementation

### Conversation Generation

- "DeeBoT"

  A DBT skills training virtual assistant which initiates interactions with a friendly greeting, assesses the user's emotional state, and chooses an appropriate skill to teach based on the user’s needs.

- "Patient Bot" 

  Given one of 81 pre-written starting messages capturing various challenges a skills training client may face, Patient Bot is directed via system prompt to continue the conversation from their perspective.

### Evaluation

- DBT Adherence

  Starting from a selection of the 23 standards which were were relevant in a text-only context out of the full 26 standards outlined in the DBT Adherence Checklist for Individual Therapy (DBT AC-I), we adapted each for use by GPT to provide a binary adherence rating given an input conversation.

- Skill Recommendation

  Identifies which of the 46 DBT skills listed in the DBT Skills Training Manual appeared in a conversation. After identifying the skills present, they can be compared to the set of skills that were targeted for each patient profile.

## Overview of Files

### convo_generation folder

- generate_conversations.py

  This Python script utilizes the OpenAI API through an AzureOpenAI client to simulate conversations based on seed prompts. These conversations are artificially created by creating varying 'personas' that engage with our chatbot. How this works is it first reads seed prompts that we manually created and imported into a CSV file. Next, it simulates conversations through functions created in our SimThread class, found in sim_thread.py. And lastly, it saves the conversation generated to a pickle file, specified by the user. The script also includes logging and error handling functionalities to track execution.

- sim_thread.py

  This is the SimThread class, which is designed to simulate the conversations described in the generate_conversations.py file. The class manages conversation threads between two agents, 'BerkeleyDBT' and the varying 'personas' we created from seed prompts. BerkeleyDBT and these personas will alternate responses with each other until a conversation is produced to a desired length. The class includes methods for generating responses, updating conversation statistics, and printing messages to the console or evaluation format.

### data folder

- prompts_with_ids.tsv

  This tab-separated values file lists each persona prompt with the respective DBT skills that should be applied, as well as a unique ID.

- Evaluation Prompts.tsv
  
  This tab-separated values file lists 23 DBT standards found in the DBT Adherence Checklist for Individual Therapy with its corresponding system prompt, ID, and category.

- dbt_skills_ref.csv

  This tab-separated values file lists each of the 46 skills defined in the DBT Skills Training Manual with its corresponding module focus, module name, module category, and description.  

- skill_presence_regex.json

  The json file is a collection of RegEx expression to help identify whether`` or not a specified DBT skill was mentioned in a conversation prompt.

### evaluation folder

- evaluation.py

  This python script includes the Evaluation class, which serves as an abstract base class for running the steps in the validation pipeline.
  
- adherence_evaluation.py

  This python script include the AdherenceEvaluation class, which extends the Evaluation class. It is primarily designed to evaluate the adherence of a conversation to predefined standards using the OpenAI API. It first reads the 23 DBT standards found in Evaluations_prompts.tsv. Next, it iteratively evaluates a given conversation against these 23 standards using the OpenAI chat API. Lastly, it calculates the proportion of adherence for each criterion across multiple iterations, returning the aggregated results. 

- skill_presence_evaluation.py

  This python script evaluates a conversation on whether or not a specific DBT skill was present. It returns a list of skill IDs that are present within each of the conversations.

- evaluation_pipeline.py

  This python script takes an input file with the conversations to be evaluated and runs them through one or several evaluations.

### Complete_pipeline.ipynb

  This python notebook provides a complete depiction of our pipeline process. Here, we first generate a set of conversations, then evaluate their contents through the lenses of DBT adherence and skills presence.

Copy the `env.sample` to `.env` and add the OPENAI_API_KEY.

## Contact Information

Hector Rincon - hrincon@berkeley.edu

Robert Mueller - robertmueller@berkeley.edu

Andrew Loeber - aloeber@berkeley.edu

John Van - johnvan@berkeley.edu

## Acknowledgements

Joyce Shen & Todd Holloway - Our capstone instructors for the Spring 2024 semester of UC Berkeley's MIDS program, who provided dedicated and insightful guidance from start to finish

Dr. Chris Harrison - founder of the Wise Mind Institute, consulted for his expertise on clinical delivery and Dialectical Behavioral Therapy

Inna Lin - PhD candidate at the University of Washington's Behavioral Research Lab, consulted for her expertise on behavioural LLM implementation in the mental health/DBT space
