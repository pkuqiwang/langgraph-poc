import os
import re
import json
from contextlib import asynccontextmanager
from typing import List, Tuple
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import sys
import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from demoRagAgent import (
    ToolConfig,
    create_graph,
    save_graph_visualization,
    get_llm,
    get_tools,
    Config,
    ConnectionPool,
    ConnectionPoolError,
    monitor_connection_pool,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []  # 清空默认处理器

handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes = Config.MAX_BYTES,
    backupCount = Config.BACKUP_COUNT
)

handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


def format_response(response):
    """对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出。

    Args:
        response: 输入的文本。

    Returns:
        具有清晰段落分隔的文本。
    """
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# 管理 FastAPI 应用生命周期的异步上下文管理器，负责启动和关闭时的初始化与清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, tool_config
    db_connection_pool = None
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        tools = get_tools(llm_embedding)
        tool_config = ToolConfig(tools)
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=10
        )

        try:
            db_connection_pool.open()
            logger.info("Database connection pool initialized")
            logger.debug("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open connection pool: {e}")
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        try:
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            sys.exit(1)

        save_graph_visualization(graph)

    except ConnectionPoolError as e:
        logger.error(f"Connection pool error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    yield
    if db_connection_pool and not db_connection_pool.closed:
        db_connection_pool.close()
        logger.info("Database connection pool closed")
    logger.info("The service has been shut down")

app = FastAPI(lifespan=lifespan)


async def handle_non_stream_response(user_input, graph, tool_config, config):
    """
    处理非流式响应的异步函数，生成并返回完整的响应内容。

    Args:
        user_input (str): 用户输入的内容。
        graph: 图对象，用于处理消息流。
        tool_config: 工具配置对象，包含可用工具的名称和定义。
        config (dict): 配置参数，包含线程和用户标识。

    Returns:
        JSONResponse: 包含格式化响应的 JSON 响应对象。
    """
    # 初始化 content 变量，用于存储最终响应内容
    content = None
    try:
        # 启动 graph.stream 处理用户输入，生成事件流
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # 遍历事件流中的每个事件
        for event in events:
            # 遍历事件中的所有值
            for value in event.values():
                # 检查事件值是否包含有效消息列表
                if "messages" not in value or not isinstance(value["messages"], list):
                    # 记录警告日志，跳过无效消息
                    logger.warning("No valid messages in response")
                    continue

                # 获取消息列表中的最后一条消息
                last_message = value["messages"][-1]

                # 检查消息是否包含工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # 遍历所有工具调用
                    for tool_call in last_message.tool_calls:
                        # 验证工具调用是否为字典且包含名称
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            # 记录工具调用日志
                            logger.info(f"Calling tool: {tool_call['name']}")
                    # 跳过本次循环，继续处理下一事件
                    continue

                # 检查消息是否包含内容
                if hasattr(last_message, "content"):
                    # 将消息内容赋值给 content
                    content = last_message.content

                    # 检查是否为工具输出（基于工具名称）
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        # 获取工具名称
                        tool_name = last_message.name
                        # 记录工具输出日志
                        logger.info(f"Tool Output [{tool_name}]: {content}")
                    # 处理大模型输出（非工具消息）
                    else:
                        # 记录最终响应日志
                        logger.info(f"Final Response is: {content}")
                else:
                    # 记录无内容的消息日志，跳过处理
                    logger.info("Message has no content, skipping")
    except ValueError as ve:
        # 捕获并记录值错误
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        # 捕获并记录其他未预期的异常
        logger.error(f"Error processing response: {e}")

    # 格式化响应内容，若无内容则返回默认值
    formatted_response = str(format_response(content)) if content else "No response generated"
    # 记录格式化后的响应日志
    logger.info(f"Results for Formatting: {formatted_response}")

    # 构造返回给客户端的响应对象
    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as resp_error:
        # 捕获并记录构造响应对象时的异常
        logger.error(f"Error creating response object: {resp_error}")
        # 构造错误响应对象
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content="Error generating response"),
                    finish_reason="error"
                )
            ]
        )

    # 记录发送给客户端的响应内容日志
    logger.info(f"Send response content: \n{response}")
    # 返回 JSON 格式的响应对象
    return JSONResponse(content=response.model_dump())


# 处理流式响应的异步函数，生成并返回流式数据
async def handle_stream_response(user_input, graph, config):
    """
    处理流式响应的异步函数，生成并返回流式数据。

    Args:
        user_input (str): 用户输入的内容。
        graph: 图对象，用于处理消息流。
        config (dict): 配置参数，包含线程和用户标识。

    Returns:
        StreamingResponse: 流式响应对象，媒体类型为 text/event-stream。
    """
    async def generate_stream():
        """
        内部异步生成器函数，用于产生流式响应数据。

        Yields:
            str: 流式数据块，格式为 SSE (Server-Sent Events)。

        Raises:
            Exception: 流生成过程中可能抛出的异常。
        """
        try:
            # 生成唯一的 chunk ID
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            # 调用 graph.stream 获取消息流
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            # 遍历消息流中的每个数据块
            for message_chunk, metadata in stream_data:
                try:
                    # 获取当前节点名称
                    node_name = metadata.get("langgraph_node") if metadata else None
                    # 仅处理 generate 和 agent 节点
                    if node_name in ["generate", "agent"]:
                        # 获取消息内容，默认空字符串
                        chunk = getattr(message_chunk, 'content', '')
                        # 记录流式数据块日志
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        # 产出流式数据块
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    # 记录单个数据块处理异常
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            # 产出流结束标记
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            # 记录流生成过程中的异常
            logger.error(f"Stream generation error: {stream_error}")
            # 产出错误提示
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    # 返回流式响应对象
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


async def get_dependencies() -> Tuple[any, any]:
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    try:
        graph, tool_config = dependencies
        if not request.messages or not request.messages[-1].content:
            logger.error("Invalid request: Empty or invalid messages")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"The user's user_input is: {user_input}")

        config = {
            "configurable": {
                "thread_id": f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                "user_id": getattr(request, 'userId', 'unknown')
            }
        }

        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        else:
            return await handle_non_stream_response(user_input, graph, tool_config, config)

    except Exception as e:
        logger.error(f"Error handling chat completion:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Start the server on port {Config.PORT}")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)


