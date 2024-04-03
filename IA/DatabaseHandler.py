import os
import mysql.connector
from dotenv import load_dotenv

class DatabaseHandler:
    def __init__(self):
        load_dotenv()
        self.DATABASE_USER = os.getenv("DATABASE_USER")
        self.DATABASE_NAME = os.getenv("DATABASE_NAME")
        self.DATABASE_HOST = os.getenv("DATABASE_HOST")
        self.DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
        self.mydb = None

    def connect(self):
        self.mydb = mysql.connector.connect(
            host=self.DATABASE_HOST, user=self.DATABASE_USER, password=self.DATABASE_PASSWORD, database=self.DATABASE_NAME
        )

    def disconnect(self):
        if self.mydb:
            self.mydb.close()

    def insert_record(self, tipo):
        if not self.mydb:
            self.connect()
        
        mycursor = self.mydb.cursor()

        sql = "INSERT INTO log (entrada) VALUES (%s)"
        val = (bool(tipo),)
        mycursor.execute(sql, val)

        self.mydb.commit()

        print(mycursor.rowcount, "record inserted.")

# Usage
db_handler = DatabaseHandler()
db_handler.insert_record(True)
