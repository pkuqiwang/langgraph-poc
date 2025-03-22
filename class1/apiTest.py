import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}
stream_flag = False


input_text = "200元以下，流量大的套餐有啥"
# input_text = "就上面提到的这个套餐，是多少钱"
# input_text = "你说那个10G的套餐，叫啥名字"
# input_text = "你说那个100000000G的套餐，叫啥名字"


# 封装请求的参数
data = {
    "messages": [{"role": "user", "content": input_text}],
    "stream": stream_flag,
    "userId":"456",
    "conversationId":"456"
}


if stream_flag:
    full_response = ""
    try:
        with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
            for line in response.iter_lines():
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    if not json_str:
                        logger.info(f"收到空字符串，跳过...")
                        continue
                    if json_str.startswith('{') and json_str.endswith('}'):
                        try:
                            data = json.loads(json_str)
                            if 'delta' in data['choices'][0]:
                                delta_content = data['choices'][0]['delta'].get('content', '')
                                full_response += delta_content
                                logger.info(f"流式输出，响应部分是: {delta_content}")
                            if data['choices'][0].get('finish_reason') == "stop":
                                logger.info(f"接收JSON数据结束")
                                logger.info(f"完整响应是: {full_response}")
                        except json.JSONDecodeError as e:
                            logger.info(f"JSON解析错误: {e}")
                    else:
                        logger.info(f"无效JSON格式: {json_str}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
else:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    content = response.json()['choices'][0]['message']['content']
    logger.info(f"非流式输出，响应内容是: {content}\n")