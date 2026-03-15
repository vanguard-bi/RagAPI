"""Tests for vector store factory shutdown and cleanup logic."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.vector_store import factory
from app.services.vector_store.factory import close_vector_store_connections


def test_close_vector_store_connections_mongo():
    """close_vector_store_connections closes the module-level MongoClient."""
    mock_client = MagicMock()
    factory._mongo_client = mock_client

    try:
        close_vector_store_connections()
        mock_client.close.assert_called_once()
        assert factory._mongo_client is None
    finally:
        factory._mongo_client = None


def test_close_vector_store_connections_idempotent():
    """Calling close_vector_store_connections twice is safe."""
    mock_client = MagicMock()
    factory._mongo_client = mock_client

    try:
        close_vector_store_connections()
        close_vector_store_connections()
        mock_client.close.assert_called_once()
    finally:
        factory._mongo_client = None


def test_close_vector_store_connections_no_client():
    """close_vector_store_connections is safe when no client exists."""
    factory._mongo_client = None
    close_vector_store_connections()  # Should not raise


def test_get_vector_store_atlas_mongo_closes_previous_client():
    """Calling get_vector_store twice closes the first MongoClient."""
    factory._mongo_client = None

    with patch("app.services.vector_store.factory.MongoClient") as MockMC:
        mock_client_1 = MagicMock()
        mock_client_2 = MagicMock()
        MockMC.side_effect = [mock_client_1, mock_client_2]

        mock_embeddings = MagicMock()

        with patch("app.services.vector_store.factory.AtlasMongoVector"):
            factory.get_vector_store("conn1", mock_embeddings, "coll", search_index="idx")
            assert factory._mongo_client is mock_client_1
            mock_client_1.close.assert_not_called()

            factory.get_vector_store("conn2", mock_embeddings, "coll", search_index="idx")
            mock_client_1.close.assert_called_once()
            assert factory._mongo_client is mock_client_2

    factory._mongo_client = None
