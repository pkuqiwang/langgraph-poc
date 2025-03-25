# langgraph-poc
POC for langgraph based agent from local ollama

### class 3
--launch docker compose
docker compose up -d --build

--get into postgres docker
docker exec -it postgres_db sh
--in postgres docker container
apt update
apt install -y git build-essential postgresql-server-dev-15
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

--on ubunto
sudo apt install libpq-dev

