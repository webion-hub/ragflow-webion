CREATE DATABASE IF NOT EXISTS rag_flow;
USE rag_flow;
UPDATE mysql.user SET Host = '%' WHERE User = 'root'
FLUSH PRIVILEGES