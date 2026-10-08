import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

conn = psycopg.connect(database_url)

print("Connected to PostgreSQL!")

conn.close()