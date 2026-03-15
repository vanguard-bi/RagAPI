import logging
from typing import Optional

from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .atlas_mongo_vector import AtlasMongoVector

logger = logging.getLogger(__name__)

# Holds the MongoClient so it can be closed on shutdown.
_mongo_client: Optional[MongoClient] = None


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    search_index: Optional[str] = None,
    db_name: Optional[str] = None,
):
    global _mongo_client

    if _mongo_client is not None:
        _mongo_client.close()
    _mongo_client = MongoClient(connection_string)
    if db_name:
        mongo_db = _mongo_client.get_database(db_name)
    else:
        mongo_db = _mongo_client.get_default_database()
    mong_collection = mongo_db[collection_name]
    return AtlasMongoVector(
        collection=mong_collection, embedding=embeddings, index_name=search_index
    )


def close_vector_store_connections() -> None:
    global _mongo_client

    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB client closed")
        except Exception as e:
            logger.warning("Failed to close MongoDB client: %s", e)
        finally:
            _mongo_client = None
