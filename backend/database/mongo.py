from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGO_URI"))
db = client.credit_card_advisor
users = db.users
