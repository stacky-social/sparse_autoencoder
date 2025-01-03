import json
import os
import csv
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import concurrent.futures

from embeddings import EmbeddingsDataset
from topk_sae import FastAutoencoder

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
import datetime
import tenacity

load_dotenv()

# Constants
now = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

DATA_DIR = Path(os.environ.get("EMBEDDINGS_FOLDER"))
SAE_DATA_DIR = Path(os.environ.get("SAE_DATA_FOLDER"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_FOLDER")) / "neuron_analyzer"
OUTPUT_FILE = OUTPUT_DIR / f"feature_analysis_results_{now}.json"

SAVE_INTERVAL = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a folder for logs if it does not exist
if not os.path.exists(os.getenv("LOG_FOLDER")):
    os.makedirs(os.getenv("LOG_FOLDER"))

logger = logging.getLogger("sae_logger")
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(os.getenv("LOG_FOLDER"), f"{now}.log"),
    filemode='w')
logger.setLevel(logging.DEBUG)

@dataclass
class Feature:
    index: int
    label: str
    reasoning: str
    top_posts: list
    zero_posts: list
    f1: float
    pearson_correlation: float
    density: float

class BatchNeuronAnalyzer:
    AUTOINTERP_PROMPT = """ 
You are a meticulous AI and social media researcher conducting an important investigation into a certain neuron in a language model trained on social media posts. Your task is to figure out what sort of behaviour this neuron is responsible for -- namely, on what general concepts, features, topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of social media posts that activate the neuron, along with a number measuring how much it was activated (these number's absolute scale is meaningless, but the relative scale may be important). This means there is some feature, topic or concept in this text that 'excites' this neuron.

ZERO_ACTIVATING_EXAMPLES_DESCRIPTION
You will also be given several examples of posts that don't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks. Be concise, and information dense. Don't waste a single word of reasoning.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, and features that all examples share in common. These can be abstract but will need to be specific - remember, all of the text comes from social media posts, so these need to be highly specific public topics of debate. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. However, the only requirement is that all posts contain this feature.
Step 2: Based on the zero activating examples, rule out any of the topics/concepts/features listed above that are in the zero-activating examples. Systematically go through your list above.
Step 3: Based on the above two steps, perform a thorough analysis of which feature, concept or topic, at what level of granularity, is likely to activate this neuron. Use Occam's razor, the simplest explanation possible, as long as it fits the provided evidence. Opt for general concepts, features and topics. Be highly rational and analytical here.
Step 4: Based on step 4, summarise this concept in 1-8 words, in the form "FINAL: <explanation>". Do NOT return anything after this. 

Here are the max-activating examples:

{max_activating_examples}

Here are the zero-activating examples:

{zero_activating_examples}

Work through the steps thoroughly and analytically to interpret our neuron.
"""

    PREDICTION_BASE_PROMPT = """
You are an AI expert that is predicting which social media posts will activate a certain neuron in a language model trained on social media posts. 
Your task is to predict which of the following posts will activate the neuron the most. Here's how you'll complete the task:

INPUT_DESCRIPTION:
You will be given the description of the type of social media posts on which the neuron activates. This description will be short.

You will then be given a post. Based on the concept of the post, you will predict whether the neuron will activate or not.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the description of the type of social media posts on which the neuron activates, reason step by step about whether the neuron will activate on this post or not. Be highly rational and analytical here. The post may be nuanced - it may contain topics/concepts close to the neuron description, but not exact. In this case, reason thoroughly and use your best judgement.
Step 2: Based on the above step, predict whether the neuron will activate on this post or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1 if you're certain it will activate because it contains topics/concepts that match the description exactly, 0 if you're highly uncertain). If you predict it will not activate, give a confidence score from -1 to 0.
Step 3: Provide the final prediction in the form "PREDICTION: <number>". Do NOT return anything after this.

Here is the description/interpretation of the type of social media posts on which the neuron activates:
{description}

Here is the post to predict:
{post}

Work through the steps thoroughly and analytically to predict whether the neuron will activate on this social media post.
"""

    def __init__(self, model_name):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.topk_indices, self.topk_values = self.load_sae_data(model_name)
        self.embeddings, self.metadata = self.load_embeddings()

    def load_sae_data(self, model_name) -> Tuple[np.ndarray, np.ndarray]:
        topk_indices = np.load(SAE_DATA_DIR / f"{model_name}_topk_indices.npy") # (data_size, k)
        topk_values = np.load(SAE_DATA_DIR / f"{model_name}_topk_values.npy")
        logger.info(f"loaded {model_name}_topk_indices.npy and {model_name}_topk_values.npy")
        print(topk_indices.shape, topk_values.shape)
        return topk_indices, topk_values

    def load_embeddings(self) -> Tuple[torch.tensor, pd.DataFrame]:
        logger.info("loading embeddings...")
        path = DATA_DIR / "full"
        dataset = EmbeddingsDataset()
        dataset.load_all_embeddings(dir=path, from_dir=True)    
        logger.info(f"Successfully loaded {len(dataset.embeddings)} embeddings")
        return dataset.embeddings, dataset.metadata

    def get_feature_activations(self, feature_index: int, m: int, min_length: int = 100) -> Tuple[List[Tuple], List[Tuple]]:
        """
            Args:
                feature_index: the index of the SAE feature to analyze
                m: the number of documents to return
                min_length: the minimum length of document to include
        """
        logger.info(f"getting {m} feature activations for feature {feature_index}...")
        doc_ids = self.metadata['id']
        posts = self.metadata['text']
        
        feature_mask = self.topk_indices == feature_index # (data_size, k)?
        activated_indices = np.where(feature_mask.any(axis=1))[0] # (activated_samples,) a list of examples which activated this feature
        activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1) # (data_size,) a list of activation values for each example, 0 if this example did not activate this feature
        
        sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])] # (activated_samples,)
        
        top_m_posts = []
        top_m_indices = []
        for i in sorted_activated_indices: # in order of most to least activating examples
            if len(posts[i]) > min_length:
                top_m_posts.append((doc_ids[i], posts[i], activation_values[i]))
                top_m_indices.append(i)
            if len(top_m_posts) == m:
                break
        
        zero_activation_indices = np.where(~feature_mask.any(axis=1))[0] # (activated_samples,) a list of examples which don't activate this feature at all
        zero_activation_samples = []
        
        active_embedding = np.array([self.embeddings[i] for i in top_m_indices]).mean(axis=0) # (1536,)
        cosine_similarities = np.dot(active_embedding, self.embeddings[zero_activation_indices].T) 
        cosine_pairs = [(index, cosine_similarities[i]) for i, index in enumerate(zero_activation_indices)]
        cosine_pairs.sort(key=lambda x: -x[1])
        
        for i, cosine_sim in cosine_pairs: # find the zero activating examples which are closest to the average embedding of the top m examples
            if len(posts[i]) > min_length:
                zero_activation_samples.append((doc_ids[i], posts[i], 0))
            if len(zero_activation_samples) == m:
                break
        
        logger.info(f"returning {len(top_m_posts)} top_m_posts and {len(zero_activation_samples)} zero_activation_samples")
        return top_m_posts, zero_activation_samples

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception),
        before=tenacity.before_log(logger, logging.INFO),  # Logs before each retry
        after=tenacity.after_log(logger, logging.INFO)     # Logs after each retry
    )
    def generate_interpretation(self, top_posts: List[Tuple], zero_posts: List[Tuple]) -> str:
        logger.debug("generating an interpretation...")
        max_activating_examples = "\n\n------------------------\n".join([f"Activation:{activation:.3f}\n{post}" for _, post, activation in top_posts])
        zero_activating_examples = "\n\n------------------------\n".join([post for _, post, _ in zero_posts])
        
        prompt = self.AUTOINTERP_PROMPT.format(
            max_activating_examples=max_activating_examples,
            zero_activating_examples=zero_activating_examples
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        logger.debug(f"Received a response {response}")
        
        return response.choices[0].message.content

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception),
        before=tenacity.before_log(logger, logging.INFO),  # Logs before each retry
        after=tenacity.after_log(logger, logging.INFO)     # Logs after each retry
    )
    def predict_activation(self, interpretation: str, post: str) -> float:
        logger.debug("in predict_activation. Formatting prompt...")
        prompt = self.PREDICTION_BASE_PROMPT.format(description=interpretation, post=post)
        logger.debug("in predict_activation. generating response")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        logger.debug(f"in predict_activation. received response {response}")
        response_text = response.choices[0].message.content
        logger.debug(f"response {response}")
        try:
            prediction = response_text.split("PREDICTION:")[1].strip()
            return float(prediction.replace("*", ""))
        except Exception:
            return 0.0

    def predict_activations(self, interpretation: str, posts: List[str]) -> List[float]:
        logger.info(f"predicting activations on {len(posts)} posts...")
        return [self.predict_activation(interpretation, post) for post in posts]

    @staticmethod
    def evaluate_predictions(ground_truth: List[int], predictions: List[float]) -> Tuple[float, float]:
        correlation, _ = pearsonr(ground_truth, predictions)
        binary_predictions = [1 if p > 0 else 0 for p in predictions]
        f1 = f1_score(ground_truth, binary_predictions)
        return correlation, f1

    def analyze_feature(self, feature_index: int, num_samples: int) -> Feature:
        logger.info(f"analyzing feature {feature_index}")
        
        #return format: ((doc_id, text, activation_amount), (doc_id, text, 0))
        top_posts, zero_posts = self.get_feature_activations(feature_index, num_samples)
        top_posts_result = [{"id": int(doc_id), "text": text, "activation": float(activation)} for doc_id, text, activation in top_posts]
        zero_posts_result = [{"id": int(doc_id), "text": text, "activation": float(activation)} for doc_id, text, activation in zero_posts]
        
        logger.debug(f"generating interpretation for feature {feature_index}...")
        interpretation_full = self.generate_interpretation(top_posts, zero_posts)
        logger.debug(f"generated interpretation for feature {feature_index}")
        interpretation = interpretation_full.split("FINAL:")[1].strip()
        
        logger.debug(f"creating test_posts+ground_truth for feature {feature_index}")
        num_test_samples = 3
        test_posts = [post for _, post, _ in top_posts[-num_test_samples:] + zero_posts[-num_test_samples:]]
        ground_truth = [1] * num_test_samples + [0] * num_test_samples

        logger.debug(f"predicting activations for feature {feature_index}...")
        predictions = self.predict_activations(interpretation, test_posts)
        logger.debug(f"evaluating predictions for feature {feature_index}...")
        correlation, f1 = self.evaluate_predictions(ground_truth, predictions)

        density = (self.topk_indices == feature_index).any(axis=1).mean()

        logger.debug(f"returning feature {feature_index}")
        return Feature(
            index=feature_index, # the index of the feature
            label=interpretation, # the final LLM interpretation of the feature's label
            reasoning=interpretation_full, # the full LLM reasoning
            top_posts=top_posts_result,
            zero_posts=zero_posts_result,
            f1=f1, # the f1 score of predicting whether or not the given top posts will activate the neuron
            pearson_correlation=correlation, # the pearson correlation between the actual sample post activations and the predictions
            density=density # average number of data points which activated this feature
        )

def save_results(results: List[Dict], filename: Path):
    logger.info(f"saving results to {filename}")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename: Path) -> List[Dict]:
    if filename.exists():
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def convert_to_csv(filename: Path) -> None:
    """
    THIS IS STILL BROKEN
    """
    results = load_results(filename)

    sample = results[0]
    num_top_posts = len(sample["top_posts"])
    top_post_labels_interim = list(zip([f"top_post_{idx}_id" for idx in range(num_top_posts)], \
                        [f"top_post_{idx}_text" for idx in range(num_top_posts)], \
                        [f"top_post_{idx}_activation" for idx in range(num_top_posts)]))
    top_post_labels = []
    for tup in top_post_labels_interim:
        top_post_labels.extend(list(tup))

    # print(top_post_labels)
    num_zero_posts = len(sample["zero_posts"])
    zero_post_labels_interim = list(zip([f"zero_post_{idx}_id" for idx in range(num_zero_posts)], \
                        [f"zero_post_{idx}_text" for idx in range(num_zero_posts)], \
                        [f"zero_post_{idx}_activation" for idx in range(num_zero_posts)]))
    zero_post_labels = []
    for tup in zero_post_labels_interim:
        zero_post_labels.extend(list(tup))
        
    # print(zero_post_labels)
    header = "\t".join(["index", "label", "reasoning"] + top_post_labels + zero_post_labels + ["f1", "pearson_correlation", "density"])
    header += "\n"
    parent = filename.parent #prefix folders
    stem = filename.stem  # filename without file suffix

    filepath = parent / f"{stem}.tsv"
    logger.info(f"saving to tsv file: {filepath}")
    with open(f"{filepath}", "w") as f:
        f.write(header)
        for feature in results:
            line = ""
            line += f"{feature['index']}\t"
            line += f"{feature['label']}\t"
            cleaned_reasoning = feature['reasoning'].replace('\n', ' ').replace('\t', ' ')
            line += f"{cleaned_reasoning}\t"

            for idx, top_post in enumerate(feature["top_posts"]):
                line += f"{top_post['id']}\t"
                # text = re.sub('[\n\t]*',' ',top_post["text"]) # remove new lines and tabs from the text for display purposes
                cleaned_text = top_post["text"].replace("\n", "").replace("\t", "")
                # output   = re.sub(r"[\n\t\s]*", "", myString)
                line += f"{cleaned_text}\t"
                line += f"{top_post['activation']}\t"
                
            for idx, zero_post in enumerate(feature["zero_posts"]):
                line += f"{zero_post['id']}\t"
                # text = re.sub('[\n\t]*',' ',zero_post["text"]) # remove new lines and tabs from the text for display purposes
                cleaned_text = zero_post["text"].replace("\n", "").replace("\t", "")
                line += f"{cleaned_text}\t"
                line += f"{zero_post['activation']}\t"

            line += f"{feature['f1']}\t"
            line += f"{feature['pearson_correlation']}\t"
            line += f"{feature['density']}\t"
            line += "\n"
            f.write(line)

def convert_to_markdown(filename: Path) -> None:
    """
    THIS ONE WORKS. USE THIS TO EXPORT RESULTS
    """
    results = load_results(filename)
    parent = filename.parent #prefix folders
    stem = filename.stem  # filename without file suffix

    filepath = parent / f"{stem}.md"
    logger.info(f"saving to md file: {filepath}")
    with open(f"{filepath}", "w") as f:
        f.write("# Results\n\n")
        for result in results:
            f.write(f"## Feature {result['index']}: {result['label']}\n\n")
            f.write(f"### Stats\n\n")
            f.write(f"F1 score: {result['f1']}\n\n")
            f.write(f"Pearson Correlation: {result['pearson_correlation']}\n\n")
            f.write(f"Post Density: {result['density']}\n\n")
            f.write(f"### Top Posts\n\n")
            for post in result['top_posts']:
                f.write(f"#### Top Post {post['id']}: Activation {post['activation']}\n\n")
                f.write(f"{post['text']}\n\n")

            f.write(f"### Zero Posts\n\n")
            for post in result['zero_posts']:
                f.write(f"#### Zero Post {post['id']}: Activation {post['activation']}\n\n")
                f.write(f"{post['text']}\n\n")
            
            f.write(f"### Explanation\n\n")
            f.write(f"{result['reasoning']}\n\n")
    

def main():
    logger.info("Creating Neuron Analyzer...")
    model_name = "k64_ndirs9216_auxk128_full_initsample10k"

    # === WARNING: THESE NUMBERS DICTATE THE COST OF RUNNING THIS CODE === # 
    num_features = 10 # number of SAE features to analyze
    num_samples = 10 # number of activating examples to base analysis on
    # ==================================================================== #

    analyzer = BatchNeuronAnalyzer(model_name)


    logger.info(f"Num features: {num_features}, num_samples: {num_samples}")

    # Load existing results and determine the starting point
    results = load_results(OUTPUT_FILE)
    start_index = max([feature['index'] for feature in results], default=-1) + 1
    print(f"Starting analysis from feature {start_index}...")
    logger.info(f"Starting analysis from feature {start_index}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(analyzer.analyze_feature, i, num_samples): i 
                           for i in range(start_index, num_features)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                           total=num_features - start_index, 
                           desc="Analysing features"):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                results.append(asdict(feature))
                
                # Save checkpoint
                if len(results) % SAVE_INTERVAL == 0:
                    save_results(results, OUTPUT_FILE)
                    print(f"Checkpoint saved. Processed {len(results)} features.")
                    logger.info(f"Checkpoint saved. Processed {len(results)} features.")
                
            except Exception as exc:
                print(f"Feature {feature_index} generated an exception: {exc}")
                logger.error(f"Feature {feature_index} generated an exception: {exc}")

    save_results(results, OUTPUT_FILE)
    print(f"Analysis complete. Results saved to {OUTPUT_FILE}")
    logger.info(f"Analysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":

    # Create a folder for outputs if it does not exist
    print(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Hello world")

    # main()
    result_path = OUTPUT_DIR / "feature_analysis_results_2024-12-26T15_06_48.json"
    # convert_to_csv(result_path)
    convert_to_markdown(result_path)