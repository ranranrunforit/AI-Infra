#!/bin/bash
set -e

# Create multiple databases
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE airflow;
EOSQL