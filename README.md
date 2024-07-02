# Project Title: MediBot 🧑🏽‍⚕️

## Description

MediBot is an intelligent chatbot designed to provide personalized information and support related to mental health and general wellness. Powered by advanced natural language processing (NLP) techniques, including Hugging Face transformers and FAISS for efficient document retrieval, MediBot aims to offer insightful responses and guidance to users seeking information on various health topics.

## Overview

MediBot utilizes cutting-edge technologies to enhance user interaction and support:
- **Conversational AI:** Integrates Hugging Face transformers to understand and respond to user queries effectively.
- **Document Retrieval:** Utilizes FAISS vector stores for quick and accurate retrieval of relevant health information from a collection of documents.
- **Memory Feature:** Maintains a conversation history to provide contextually aware responses and enhance user experience over multiple interactions.

## Models Used

MediBot employs the following models:
- **LLaMA-2-7B Model:** Filename `llama-2-7b-chat.ggmlv3.q4_0.bin`. This model is used for generating conversational responses.
- **[Sentence-Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):** `sentence-transformers/all-MiniLM-L6-v2`. This model enhances the understanding and representation of user queries for effective interaction.

## Installation

List the steps to install any dependencies and set up your environment. For example:

```bash
pip install -r requirements.txt
```

## Usage

Explain how to run your application or use your project. Include any necessary commands or configuration details. For example:

```bash
streamlit run app.py
```

## Features

Highlight the main features or functionalities of your project. For example:
- Conversational AI powered by Hugging Face transformers.
- Document retrieval using FAISS vector stores.
- Memory feature to maintain chat history.

## Dataset

The dataset used in this project is sourced from a medical book.

