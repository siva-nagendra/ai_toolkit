# AI Toolkit 🌟

AI Toolkit is your one-stop solution for integrating, training, and utilizing large language models and AI agents. Tailored to facilitate seamless workflows, it leverages the prowess of artificial intelligence to amplify your productivity and innovation.

[![Python package](https://img.shields.io/pypi/v/ai_toolkit)](https://pypi.org/project/ai_toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/ai_toolkit.svg?maxAge=2592000)](https://pypi.python.org/pypi/ai_toolkit/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ai_toolkit)](https://pypi.org/project/ai_toolkit/)
[![License: MIT](https://img.shields.io/github/license/siva-nagendra/ai_toolkit)](https://github.com/siva-nagendra/ai_toolkit/blob/main/LICENSE)

> AI Toolkit is open for collaboration! Join us to enhance AI Toolkit by participating in [discussions](https://github.com/siva-nagendra/ai_toolkit/discussions), opening [issues](https://github.com/siva-nagendra/ai_toolkit/issues/new/choose), or submitting [PRs](https://github.com/siva-nagendra/ai_toolkit/pulls).

## 🚀 Features

1. **Load Multiple Models** :gear: Easily load various models from Huggingface and OpenAI, integrating the power of AI seamlessly.
2. **Train Your Models** :brain: Train your models with diverse file formats seamlessly integrated into the application.
3. **Chat with LLMs** :speech_balloon: Engage with Large Language Models via the intuitive Chainlit Chatbot UI.
4. **Web Surfing to Train Capacity** :globe_with_meridians: (Under Development): Extract training data from the web for more nuanced model training.
5. **AI Agents for Task Automation** :robot: (Under Development): Utilize AI agents to automate specific tasks and workflows.

## 🔧 Installation

To get started with AI Toolkit, ensure you have Python 3.11 installed. Follow the steps below to install all the necessary dependencies:

```sh
pip install -r requirements.txt
```

## 💻 Usage

```sh
# To load the vector database
ai_toolkit load --db-path <path_to_vector_database>

# To train the model with the specified dataset
ai_toolkit train --dataset-path <path_to_dataset> --db-path <path_to_vector_database>
```

## ⚙️ Environment Variables

1. `DB_FAISS_PATH`: Path to the vector database. 
2. `DATASET_PATH`: Path to the dataset.

## 🤝 Contributions

We warmly welcome community contributions! Feel free to enhance AI Toolkit's capabilities through issues and pull requests.

## 📜 License

This project is licensed under the [MIT License](https://github.com/siva-nagendra/ai_toolkit/blob/main/LICENSE) - see the [LICENSE](https://github.com/siva-nagendra/ai_toolkit/blob/main/LICENSE) file for details.
