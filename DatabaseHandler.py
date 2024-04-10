import os
from supabase import create_client, Client
from dotenv import load_dotenv

class DatabaseHandler:
    def __init__(self):
        load_dotenv()
        self.URL_SUPABASE = os.getenv("URL_SUPABASE")
        self.DATABASE_APIKEY = os.getenv("DATABASE_APIKEY")
        self.mydb = None

    def connect(self):
        self.mydb: Client = create_client(self.URL_SUPABASE, self.DATABASE_APIKEY)

    def disconnect(self):
        if self.mydb:
            self.mydb.close()

    def insert_record(self, tipo, quantidade):
        self.connect()
        data, count = self.mydb.table('log').insert({"entrada": bool(tipo), "lotacao": quantidade}).execute()
        print("1 record inserted.")