from pymilvus import MilvusClient, Collection, connections
from dotenv import load_dotenv
import os
from torch.utils.data import Dataset, DataLoader
import torch
import time
import random
import logging
import datetime
import json

import cProfile

load_dotenv()
logger = logging.getLogger(__name__)
# Create a folder for logs if it does not exist
if not os.path.exists(os.getenv("LOG_FOLDER")):
    os.makedirs(os.getenv("LOG_FOLDER"))

# Create a folder for embeddings if it does not exist
if not os.path.exists(os.getenv("EMBEDDINGS_FOLDER")):
    os.makedirs(os.getenv("EMBEDDINGS_FOLDER"))

now = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(os.getenv("LOG_FOLDER"), f"{now}.log"),
    filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NoEmbeddingsError(Exception):
    def __init__(self, message):            
        super().__init__(message)

class EmbeddingsDataset(Dataset):

    def __init__(self, collection_name="mock_mastadon", 
                partitions=["_default", "112734616419053181", "112735699993230698", "nyt_injected", "stackyredditcom"]):
        """
        Args:
            collection_name (string): Name of the collection in Milvus
            partitions (list): List of partitions within the collection
        """
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_token = os.getenv("MILVUS_TOKEN")
        self.collection_name = collection_name
        self.partitions = partitions
        self.embeddings = None # this gets populated via load_all_embeddings

        self.client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self.connection = connections.connect(uri=milvus_uri, token=milvus_token)
        self.collection = Collection(collection_name)
        
        self.load_all_partitions()

    def load_all_embeddings(self, dir=os.environ.get("EMBEDDINGS_FOLDER"), from_dir=False):
        """
            The entry point for loading all embeddings from Milvus
            Args:
                dir (string): Path to a folder where the embeddings are stored. 
                from_dir (bool): If True, load embeddings from the given directory. 
                    If False, load embeddings from Milvus, and save them to the given directory, and then load them                    
        """
        if from_dir == False:
            self.save_embeddings_from_milvus(dir=dir)
        
        self.embeddings = self.load_all_embeddings_from_folder(dir=dir)

        return self.embeddings
    
    def load_all_embeddings_from_folder(self, dir=os.environ.get("EMBEDDINGS_FOLDER")):
        """
            Load all embeddings from a the given directory
            Args:
                dir (string): Path to a folder where embeddings are stored.
        """
        #iterate over all files in this directory, and stack them into one big tensor of embeddings
        embeddings = None
        for filename in os.listdir(dir):
            if filename.endswith(".pt"):
                filepath = os.path.join(dir, filename)
                with open(filepath, 'rb') as f:
                    embeddings_tensor = torch.load(f)
                    logger.info(f"Loaded tensor of size {embeddings_tensor.size()} from file {filepath}")
                    if embeddings is None:
                        embeddings = embeddings_tensor
                    else:
                        embeddings = torch.cat((embeddings, embeddings_tensor), dim=0)
        
        logger.info(f"Final embeddings tensor loaded {embeddings.size()}")
        return embeddings

    def save_embeddings_from_milvus(self, dir=os.environ.get("EMBEDDINGS_FOLDER")):
        """
            Save embeddings from Milvus. This is done in batches, and each batch is turned into a 
            tensor and saved to a file in the given directory
            Args:
                dir (string): Path to a folder where the embeddings will be saved
        """

        iterator = self.collection.query_iterator(
            batch_size=10, # for testing
            # batch_size=500,
            limit=50, # for testing
            output_fields=["id", "text_vector", "text"]
        )

        start = 0 # for file naming
        end = 0 # for file naming
        total = 0 # for logging
        while True:
            result = iterator.next()
            num_returned = len(result)
            total += num_returned
            logger.info(f"queried {num_returned} embeddings")
            end += num_returned

            if not result:
                iterator.close()
                break
            else:
                # turn the text_vectors into a 2D tensor of size (limit x 1536)
                logger.info("Converting embeddings to tensor...")
                embeddings_tensor = torch.tensor([x['text_vector'] for x in result], dtype=torch.float32)        
                logger.info(f"{embeddings_tensor.size()}")
                filepath = os.path.join(dir, f'{start}_{end}.pt')
                with open(filepath, 'wb') as f:
                    torch.save(embeddings_tensor, f)
                logger.info(f"Embeddings saved to file {filepath}")
            
            start = end+1
                    
        logger.info(f"Total number of embeddings saved: {total}")
    
    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.embeddings[idx]

    def load_partition(self, partition_names):
        self.client.load_partitions(self.collection_name, partition_names)
    
    def release_partition(self, partition_names):
        self.client.release_partitions(self.collection_name, partition_names)
    
    def load_all_partitions(self):
        for partition in self.partitions:
            logger.info(f"Loading partition {partition}")
            self.load_partition(partition)
            logger.info(self.get_partition_stats(partition))
        
        logger.info("All partitions loaded")

    def release_collection(self):
        self.client.release_collection(self.collection_name)

    def get_partition_stats(self, partition_names):
        return self.client.get_partition_stats(self.collection_name, partition_names)


if __name__ == "__main__":
    # cProfile.run('dataset = EmbeddingsDataset(save_to_file=True)', sort='cumtime')
    
    dataset = EmbeddingsDataset()

    # Create a folder for embeddings if it does not exist
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), f"test")
    if not os.path.exists(path):
        os.makedirs(path)

    dataset.load_all_embeddings(dir=path, from_dir=True)
    dataset.release_collection()