from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGODB_URI")

client = AsyncIOMotorClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client["devops_ai"]
