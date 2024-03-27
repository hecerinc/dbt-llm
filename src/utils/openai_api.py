from openai import AzureOpenAI
import os, sys

sys.path.append(os.path.join(sys.path[0], '../..'))
import time
import re

CHAT_ERROR_TEXT = "Error encountered. LLM could not return result for the associated query."

def get_openai_response(client: AzureOpenAI, messages, model: str):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    print('OPENAI', response)
    return response

def get_openai_response_content(client: AzureOpenAI, messages, model: str):
    num_tries = 3
    while num_tries > 0:
        try:
            response = get_openai_response(client, messages, model)
            return response.choices[0].message.content
        except Exception as e:
            sleep_time = 5.0
            e_str = str(e).lower()
            if 'content management policy' in e_str:
                completion = "LLM did not return result due to triggering Azure OpenAI's content policy."
                return completion
            elif 'rate limit' in e_str:
                sleep_time_str = re.findall(r'\d+', str(e))[0]
                sleep_time = float(sleep_time_str) + 1.0
            num_tries -= 1
            time.sleep(sleep_time)
    return CHAT_ERROR_TEXT

