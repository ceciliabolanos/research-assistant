# ScientificGPT

## Introduction

In our academic journey, we often encounter research papers that reference specific code implementations. More than once, we've had to delve into these implementations to understand details not explicitly mentioned in the text. This experience inspired us to develop a scientific agent capable of bridging the gap between research papers and their associated code. This repository contains the initial MVP of such an agent, currently limited to code access.

## Overview

ScientificGPT is a Retrieval Augmented Generation (RAG) system that responds to natural language queries with the most relevant code snippets. It utilizes various components to perform tasks such as searching for code by similarity or giving directory trees from GitHub. The Searcher class uses a fine-tuned UnixCoder model, originally from Microsoft for CodeSearch, to align docstring embeddings with code embeddings. Given that a human query might not directly match the trained docstrings, we further fine-tuned the Mistral model on a custom dataset of query-docstring pairs to better translate queries into functional docstrings. Users have the option to use this enhanced Mistral model, which, if selected, pre-processes the query into a docstring before invoking the similarity search function.

## Directory Structure

- `./databases`: Contains FAISS databases with embeddings for previously queried GitHub repositories, eliminating the need to regenerate embeddings.
- `./models`: Stores scripts for UnixCoder and Mistral models.
- `./parse_code`: Scripts for parsing GitHub content to JSON and converting code strings into sequences as per the UnixCoder paper.
- `./parse_pdf`: Tools for converting PDF papers to JSON (not yet integrated into the agent).
- `./retrieval`: Contains the retrieval logic (ChatGPT) that interfaces with the searcher to handle queries.

## Usage

A usage example is provided in a Jupyter notebook available on this repository and can be open on Colab, which supports GPU execution necessary for using Mistral.

Alternatively, follow these setup steps:

```bash
pip install -r requirements.txt
pip install -i https://pypi.org/simple/ bitsandbytes
pip install accelerate
pip install bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git@main
pip install -q -U git+https://github.com/huggingface/peft.git
```

To use the system, configure the following parameters:

```bash
github_url = 'path to github repo'
unixcoder_path = 'path to unixcoder fine-tuned'
mistral = 'wants to use mistral? yes or no'
chat_model = "choice of GPT model for retrieval"
```

Example command:

```bash
python main.py --github_url "https://github.com/ankitapasad/layerwise-analysis.git" --model_path '/content/drive/My Drive/unixcoder-ft.bin' --mistral 'yes' --chat_model "gpt-3.5-turbo-0125"
```

This command initiates a session with an assistant that can answer queries about the GitHub repository, returning functions that closely match the query.

## Future Work

We plan to add more capabilities to the searcher to handle more specific questions about code and to integrate paper content into the information available to the searcher.


## Para que funcione pdf to json: 
   Linux:
   - wget https://github.com/kermitt2/grobid/archive/0.7.2.zip
   - unzip 0.7.2.zip
   - cd grobid-0.7.2/
   - ./gradlew clean install



  
