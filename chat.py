import os
import pickle
from typing import Any, Dict, List

import autogen
import numpy as np
import pandas as pd
from autogen import AssistantAgent, ConversableAgent, UserProxyAgent, register_function
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_CONFIG = [
    {
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
]

BASE_CONFIG = {
    "config_list": OPENAI_CONFIG,
    "temperature": 0,
    "timeout": 60,
    "seed": 42,
}


def predict_startup_success(
    age: float,
    country_code: str,
    category_code: str,
    num_acquisitions: int,
    funding_amount: float,
    num_funding_rounds: int,
    num_milestones: int,
    investment_rounds: int,
    invested_companies: int,
    relationships: int,
    received_financial_investment: int,
) -> str:
    with open("models/startup_success_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open("models/country_encoder.pkl", "rb") as f:
        country_encoder = pickle.load(f)

    with open("models/category_encoder.pkl", "rb") as f:
        category_encoder = pickle.load(f)

    try:
        encoded_country = country_encoder.transform([country_code])[0]
    except:
        encoded_country = country_encoder.transform(["UNKNOWN"])[0]

    try:
        encoded_category = category_encoder.transform([category_code])[0]
    except:
        encoded_category = category_encoder.transform(["unknown"])[0]

    input_dict = {
        "age": age,
        "country_code": encoded_country,
        "category_code": encoded_category,
        "num_acquisitions_made": num_acquisitions,
        "log_funding": np.log1p(funding_amount),
        "num_funding_rounds": num_funding_rounds,
        "num_milestones": num_milestones,
        "investment_rounds": investment_rounds,
        "invested_companies": invested_companies,
        "relationships": relationships,
        "received_financial_investment": received_financial_investment,
    }
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    result = f"Predicted outcome: {prediction}\n\nProbabilities:\n"
    for outcome, prob in zip(model.classes_, probabilities):
        result += f"{outcome}: {prob:.2%}\n"

    return result


VC_PROMPT = """You are a friendly and knowledgeable Venture Capital analyst. Your goal is to gather information about a startup to predict its success potential. Interact with the user in a conversational manner to collect all necessary information for the prediction model. You may ask multiple questions in a single message to fast-track the process.

You need to collect the following information, but do it naturally through conversation, not as a form:
1. Startup age in years
2. Country code (two-letter code like US, GB, etc.)
3. Category code (like software, biotech, mobile, etc.)
4. Number of acquisitions made
5. Total funding amount received (in USD)
6. Number of funding rounds completed
7. Number of milestones achieved
8. Number of investment rounds participated in
9. Number of companies invested in
10. Number of business relationships
11. Whether they've received investment from financial organizations (0 or 1)

Once you have all the information, use the predict_startup_success function to make a prediction.

Be conversational and explain any terms the user might not understand. Ask follow-up questions when responses are unclear. If a user is unsure about exact numbers, help them estimate based on their knowledge.

After getting a prediction, explain the results in a clear, friendly manner and offer insights based on the probabilities.

YOUR KEY TASK IS TO ENSURE YOU GET ALL REQUIRED INFORMATION BEFORE MAKING A PREDICTION.

Once everything is complete, end the conversation by writing 'TERMINATE'."""

EXECUTOR_PROMPT = """You are an executor agent responsible for running the predict_startup_success function when requested by the VC Analyst. 
Your role is to:
1. Wait for function call requests from the VC Analyst
2. Execute the function with provided parameters
3. Return the results back to the VC Analyst
Do not engage in conversation - only execute functions and return results."""

llm_config = BASE_CONFIG

user = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    code_execution_config=False,
)
vc_analyst = AssistantAgent(
    name="VC_Analyst",
    system_message=VC_PROMPT,
    llm_config=llm_config,
)
executor = ConversableAgent(
    name="Executor",
    system_message=EXECUTOR_PROMPT,
    human_input_mode="NEVER",
)

register_function(
    predict_startup_success,
    caller=vc_analyst,
    executor=executor,
    name="predict_startup_success",
    description="Predict the success of a startup based on various features",
)


def is_termination_msg(msg: Dict[str, Any]) -> bool:
    return "TERMINATE" in msg.get("content", "").upper()


def start_chat():
    groupchat = autogen.GroupChat(
        agents=[user, vc_analyst, executor], messages=[], max_round=15
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=lambda msg: is_termination_msg(msg),
    )

    user.initiate_chat(
        manager, message="Hi, I'd like to analyze the potential success of my startup."
    )


if __name__ == "__main__":
    start_chat()
