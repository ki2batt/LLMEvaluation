import importlib
import os
import pytest
import requests
from langchain_community.chat_models import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

os.environ[
        ""] = ""
@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm=lang_chain_llm)
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": "How many articles are there in the Selenium webdriver python course?",
                                     "chat_history": [
                                     ]
                                 }).json()

    sample = SingleTurnSample(
        user_input='How many articles are there in the Selenium webdriver python course?',
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],

        reference="23"
    )
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7
