from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI is not set")

client = AsyncIOMotorClient(MONGO_URI)
db = client["devops_ai"]
