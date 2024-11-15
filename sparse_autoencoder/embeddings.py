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
import csv
import pandas as pd
import numpy as np

import cProfile


load_dotenv()
logger = logging.getLogger("sae_logger")
# Create a folder for logs if it does not exist
if not os.path.exists(os.getenv("LOG_FOLDER")):
    os.makedirs(os.getenv("LOG_FOLDER"))

# Create a folder for embeddings if it does not exist
if not os.path.exists(os.getenv("EMBEDDINGS_FOLDER")):
    os.makedirs(os.getenv("EMBEDDINGS_FOLDER"))

# now = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     filename=os.path.join(os.getenv("LOG_FOLDER"), f"{now}.log"),
#     filemode='w')
# logger.setLevel(logging.DEBUG)

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
        self.metadata = None # this gets populated via load_all_embeddings

        self.client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self.connection = connections.connect(uri=milvus_uri, token=milvus_token)
        self.collection = Collection(collection_name)
        
    def load_all_embeddings(self, dir=os.environ.get("EMBEDDINGS_FOLDER"), from_dir=False, batch_size=500, limit=-1, merge=True):
        """
            The entry point for loading all embeddings from Milvus
            Args:
                dir (string): Path to a folder where the embeddings are stored. 
                from_dir (bool): If True, load embeddings from the given directory. 
                    If False, load embeddings from Milvus, and save them to the given directory, and then load them
                batch_size (int): Number of embeddings to load at a time
                    only used when from_dir is False
                limit (int): Number of embeddings to load in total. -1 to return all embeddings in Milvus             
                    only used when from_dir is False    
                merge (bool): If True, merge all the embeddings into one tensor and save it to a file   
        """
        if from_dir == False:
            self.save_embeddings_from_milvus(dir=dir, batch_size=batch_size, limit=limit)
        
        self.embeddings, self.metadata = self.load_all_embeddings_from_folder(dir=dir, merge=merge)

        return self.embeddings, self.metadata
    
    def load_all_embeddings_from_folder(self, dir=os.environ.get("EMBEDDINGS_FOLDER"), merge=True):
        """
            Load all embeddings from a the given directory. Expects a collection of *.pt files, in order, containing subsets of embeddings in dir,
            as well as a metadata.csv file containing all of the metadata, in order, for the embeddings
            If a file named "all_embeddings.pt" does not exist, it will be created by stacking all the embeddings in the directory
            If a file named "all_embeddings.pt" does exist, it will be used in conjunction with metadata.csv instead of accessing each subset individually
            Args:
                dir (string): Path to a folder where embeddings are stored.
                merge (bool): If True, merge all the embeddings into one tensor and save it to a file
                    You may want to set this to False if the file would be too big to store in memory
        """
        # check if "all_embeddings.pt" exists
        all_files = os.listdir(dir)
        if "all_embeddings.pt" in all_files:
            with open(os.path.join(dir, "all_embeddings.pt"), 'rb') as f:
                embeddings = torch.load(f)
                logger.info(f"Loaded tensor of size {embeddings.size()} from file {os.path.join(dir, 'all_embeddings.pt')}")
            metadata = pd.read_csv(os.path.join(dir, "metadata.csv"))
            logger.info(f"Loaded metadata from file {os.path.join(dir, 'metadata.csv')}")
            return embeddings, metadata

        # all_embeddings.pt does not exist - go through all of the subset files and build an all_embeddings.pt
        # iterate over all files in this directory, and stack them into one big tensor of embeddings
        embeddings = None
        for filename in all_files:
            filepath = os.path.join(dir, filename)
            with open(filepath, 'rb') as f:
                if filename.endswith(".pt"): # build the tensor holding the embeddings
                    embeddings_tensor = torch.load(f)
                    logger.info(f"Loaded tensor of size {embeddings_tensor.size()} from file {filepath}")
                    if embeddings is None:
                        embeddings = embeddings_tensor
                    else:
                        embeddings = torch.cat((embeddings, embeddings_tensor), dim=0)
                
                if filename.endswith(".csv"): # create the dataframe to hold the metadata!
                    metadata = pd.read_csv(f)
                    logger.info(f"Loaded metadata from file {filepath}")

        # save embeddings into an all_embeddings.pt 
        if merge:
            with open(os.path.join(dir, "all_embeddings.pt"), 'wb') as f:
                torch.save(embeddings, f)
                logger.info(f"Final embeddings tensor saved to file {os.path.join(dir, 'all_embeddings.pt')}")

        logger.info(f"Final embeddings tensor loaded {embeddings.size()}")
        return embeddings, metadata

    def save_embeddings_from_milvus(self, dir=os.environ.get("EMBEDDINGS_FOLDER"), batch_size=500, limit=-1):
        """
            Save embeddings from Milvus. This is done in batches, and each batch is turned into a 
            tensor and saved to a file in the given directory
            Args:
                dir (string): Path to a folder where the embeddings will be saved
                batch_size (int): Number of embeddings to load at a time
                limit (int): Number of embeddings to load in total. -1 to return all embeddings in Milvus
        """

        self.load_all_partitions()

        iterator = self.collection.query_iterator(
            batch_size=batch_size,
            limit=limit, 
            output_fields=["id", "text_vector", "text"]
        )

        start = 0 # for file naming
        end = 0 # for file naming
        total = 0 # for logging
        
        #set up the metadata file
        csv_filepath = os.path.join(dir, f'metadata.csv')
        with open(csv_filepath, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text'])
        
        while True:
            result = iterator.next()
            num_returned = len(result)
            logger.info(f"queried {num_returned} embeddings")
            end += num_returned

            if not result:
                iterator.close()
                break
            else:
                # turn the text_vectors into a 2D tensor of size (limit x 1536)
                logger.info("Converting embeddings to tensor...")
                embeddings_tensor = torch.tensor([x['text_vector'] for x in result if 'text' in x], dtype=torch.float32)        
                logger.info(f"{embeddings_tensor.size()}")
                total += embeddings_tensor.size(0)
                filepath = os.path.join(dir, f'{start}_{end}.pt')
                with open(filepath, 'wb') as f:
                    torch.save(embeddings_tensor, f)
                logger.info(f"Embeddings saved to file {filepath}")

                with open(csv_filepath, mode='a') as f:
                    writer = csv.writer(f)
                    for x in result:
                        if 'text' not in x:
                            continue
                        writer.writerow([x['id'], x['text']])
            
            start = end+1
                    
        logger.info(f"Total number of embeddings saved: {total}")
        self.release_collection()
    
    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        """
            Returns the embedding and its metadata (id and post text) at the given index as a dictionary
        """

        metadata = self.metadata.iloc[idx] # this is a pandas series
        embedding = self.embeddings[idx] # this is a tensor

        res = {
            "embedding": embedding
        } 

        for key in metadata.keys():
            res[key] = metadata[key]

        return res

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

def load_test_embeddings(folder_name="test", batch_size=10, limit=50, from_dir=False, merge=True):
    """
        Load a small subset of embeddings for testing purposes
        Args:
            folder_name (string): Name of the folder within EMBEDDINGS_FOLDER where the embeddings will be stored
            batch_size (int): Number of embeddings to load at a time
            limit (int): Number of embeddings to load in total. -1 to return all embeddings in Milvus
            from_dir (bool): If True, load embeddings from the given directory.
                If False, load embeddings from Milvus, and save them to the given directory, and then load them
            merge (bool): If True, merge all the embeddings into one tensor and save it to a file
    """
    # Create a folder for embeddings if it does not exist
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = EmbeddingsDataset()
    dataset.load_all_embeddings(dir=path, from_dir=from_dir, batch_size=batch_size, limit=limit, merge=merge)
    
    # testing that everything worked
    print(dataset.metadata.head())
    print(dataset.metadata.shape)
    print(dataset.embeddings.size())

    return dataset

def load_all_embeddings(folder_name="full", batch_size=500, limit=-1, from_dir=False, merge=True):
    """
        Load all embeddings in Milvus
        Args:
            folder_name (string): Name of the folder within EMBEDDINGS_FOLDER where the embeddings will be stored
            batch_size (int): Number of embeddings to load at a time
            limit (int): Number of embeddings to load in total. -1 to return all embeddings in Milvus
            from_dir (bool): If True, load embeddings from the given directory.
                If False, load embeddings from Milvus, and save them to the given directory, and then load them
            merge (bool): If True, merge all the embeddings into one tensor and save it to a file
    """
    # Create a folder for embeddings if it does not exist
    path = os.path.join(os.environ.get("EMBEDDINGS_FOLDER"), folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = EmbeddingsDataset()
    dataset.load_all_embeddings(dir=path, from_dir=from_dir, batch_size=batch_size, limit=limit, merge=merge)
    
    # testing that everything worked
    print(dataset.metadata.head())
    print(dataset.metadata.shape)
    print(dataset.embeddings.size())

    return dataset



if __name__ == "__main__":
    # cProfile.run('dataset = EmbeddingsDataset(save_to_file=True)', sort='cumtime')
    
    #from_dir=True will load embeddings from the given directory
    #from_dir=False will pull from Milvus and make new embeddings files
    
    # dataset = load_test_embeddings(folder_name="test", from_dir=True, merge=True)

    # before uncommenting this, make sure you are on a compute node
    dataset = load_all_embeddings(folder_name="full", from_dir=True, merge=True)

    #testing the Dataset object functionality
    print(f"Dataset length: {len(dataset)}")
    for i in np.arange(0, len(dataset), 10000):
        print(f"idx {i}: {dataset[i]}")

    #sanity check these post ids:
    ids = [646107512, 111268473235526638, 112648414571546235, 112674022755516750, 112680864564528458, 112688472142170802, 112702831041142031, 112735425212348242, 112851667702767479, 112854415048218735, 112859139634975231, 112880038876951632]
    #sanity has been checked! All good :)