import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEFAULT_TEMPERATURE = 0.7


class LLMInitializationError(Exception):
    pass


def initialize_llm() -> tuple[ChatOpenAI, OllamaEmbeddings]:
    try:
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3.2:3b",
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  
            max_retries=2  
        )

        embedding = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url='http://localhost:11434',
        )
            
        return llm, embedding

    except ValueError as ve:
        raise LLMInitializationError(f"LLM itialization error: {str(ve)}")
    except Exception as e:
        raise LLMInitializationError(f"LLM itialization error:: {str(e)}")


def get_llm() -> ChatOpenAI:
    try:
        return initialize_llm()
    except LLMInitializationError as e:
        logger.warning(f"LLM itialization error: {str(e)}")
        raise 