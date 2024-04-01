import os
import mysql.connector
from dotenv import load_dotenv

def insertRecord(tipo):
    load_dotenv()
    
    DATABASE_USER = os.getenv("DATABASE_USER")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")

    mydb = mysql.connector.connect(
        host=DATABASE_HOST, user=DATABASE_USER, password=DATABASE_PASSWORD, database=DATABASE_NAME
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO log (entrada) VALUES (%s)"
    val = (bool(tipo),)
    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")

# insertRecord(True)