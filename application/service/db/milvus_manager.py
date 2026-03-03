
from re import A
from typing import List, Dict, Any, Optional
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)



class MilvusManager:
    
    def __init__(
        self, 
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        ):
        """初始化Milvus管理器"""
        self.collection = None
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type


    
    def connection_init(self):
        """连接Milvus数据库"""
        try:
            self.connections = connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )

            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

            if not utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} does not exist. Creating...")
                self.collection=Collection(
                    name=self.collection_name,
                )
                logger.info(f"Collection {self.collection_name} created successfully.")
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
            
            self.collection.load()
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            raise e
        
    
    def add_doc(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: List[Dict[str, Any]]=None,
    ):
        """添加文档到Milvus集合"""
        try:
            if metadata is None:
                metadata = [{} for _ in range(len(ids))]


            entities = [
               ids,
               embeddings,
               documents,
               [meta.get("recipe_id", "") for meta in metadatas],
               [meta.get("name", "") for meta in metadatas],
               [meta.get("category", "") for meta in metadatas],
               [meta.get("difficulty", "") for meta in metadatas],
               ]   
            self.collection.insert(
                data=entities
            )
            logger.info(f"Successfully inserted {len(ids)} documents into collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error inserting documents into Milvus: {e}")
            raise e
