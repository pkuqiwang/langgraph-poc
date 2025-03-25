import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Optional
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEFAULT_TEMPERATURE = 0.7


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm() -> tuple[ChatOpenAI, OllamaEmbeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): 'ollama'

    Returns:
        ChatOpenAI: 

    Raises:
        LLMInitializationError: 
    """
    try:
        # 创建LLM实例
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3.2:3b",
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  # 添加超时配置（秒）
            max_retries=2  # 添加重试次数
        )

        embedding = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url='http://localhost:11434',
        )
            
        logger.info(f"成功初始化 LLM and embedding")
        return llm, embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm() -> tuple[ChatOpenAI, OllamaEmbeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        ChatOpenAI: LLM实例
    """
    try:
        return initialize_llm()
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        raise  # 如果默认配置也失败，则抛出异常


# 示例使用
if __name__ == "__main__":
    try:
        # 测试不同类型的LLM初始化
        llm_openai = get_llm("openai")
        llm_qwen = get_llm("qwen")

        # 测试无效类型
        llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")