# config.py
import os

class Config:
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3

    # 数据库 URI，默认值
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012