import os
from langchain_openai import ChatOpenAI
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2:3b"
    }
}

DEFAULT_LLM_TYPE = "ollama"
DEFAULT_TEMPERATURE = 0.7


class LLMInitializationError(Exception):
    """Initialization Error"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> Optional[ChatOpenAI]:
    """
    Args:
        llm_type (str): 'ollama'

    Returns:
        ChatOpenAI: llm instance

    Raises:
        LLMInitializationError: error
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"unsupported: {llm_type}.")

        config = MODEL_CONFIGS[llm_type]

        # 创建LLM实例
        llm = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  
            max_retries=2  
        )

        logger.info(f"initialize {llm_type} LLM")
        return llm

    except ValueError as ve:
        logger.error(f"error: {str(ve)}")
        raise LLMInitializationError(f"LLM error: {str(ve)}")
    except Exception as e:
        logger.error(f"error: {str(e)}")
        raise LLMInitializationError(f"error: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    Args:
        llm_type (str): type

    Returns:
        ChatOpenAI: LLM instance
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"using default: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise 


if __name__ == "__main__":
    try:
        # test ollama
        llm_openai = get_llm("ollama")

        # test error
        llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"ending: {str(e)}")