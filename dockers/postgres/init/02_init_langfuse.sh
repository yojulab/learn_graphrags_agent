#!/bin/bash
set -e

LANGFUSE_DB="${LANGFUSE_DB_NAME:-db_langfuse}"

echo ">>> [Init] Checking and creating Langfuse database: ${LANGFUSE_DB}..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE "${LANGFUSE_DB}"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${LANGFUSE_DB}')\gexec
    GRANT ALL PRIVILEGES ON DATABASE "${LANGFUSE_DB}" TO "$POSTGRES_USER";
EOSQL
echo ">>> [Init] Langfuse database '${LANGFUSE_DB}' initialization completed."

