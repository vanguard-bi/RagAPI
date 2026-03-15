# app/utils/health.py
from app.services.mongo_client import mongo_health_check


async def is_health_ok():
    return await mongo_health_check()
