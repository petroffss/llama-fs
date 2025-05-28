from abc import ABC, abstractmethod
from typing import AsyncIterator

class BaseLLMInterface(ABC):
    """
    An abstract base class (or Protocol) defining the interface for interacting with Large Language Models (LLMs).

    This interface standardizes how different LLM implementations can be used,
    ensuring they provide core functionalities like streaming chat responses,
    generating embeddings, and providing information about the model.

    Common configuration parameters that might be passed via **kwargs in implementations include:
    - model_name (str): The specific model to use (e.g., "gpt-3.5-turbo", "claude-2").
    - api_key (str): The API key for authenticating with the LLM service.
    - temperature (float): Controls the randomness of the output. Higher values mean more random,
                         lower values mean more deterministic.
    - max_tokens (int): The maximum number of tokens to generate in a response.
    """

    @abstractmethod
    async def stream_chat(self, prompt: str, history: list = None, **kwargs) -> AsyncIterator[str]:
        """
        Streams chat responses from the LLM.

        This method should be implemented to send a prompt (and optionally, conversation history)
        to the LLM and yield response chunks as they become available.

        Args:
            prompt (str): The user's input/prompt to the LLM.
            history (list, optional): A list of previous conversation turns, often formatted as
                                      a list of dictionaries (e.g., [{"role": "user", "content": "..."},
                                      {"role": "assistant", "content": "..."}]). Defaults to None.
            **kwargs: Additional keyword arguments for LLM configuration (e.g., model_name,
                      temperature, api_key, max_tokens).

        Returns:
            AsyncIterator[str]: An asynchronous iterator yielding chunks of the LLM's response.
        """
        pass

    @abstractmethod
    async def get_embedding(self, text: str, **kwargs) -> list[float]:
        """
        Generates an embedding (vector representation) for the given text.

        Embeddings are useful for tasks like semantic search, clustering, and classification.

        Args:
            text (str): The input text to embed.
            **kwargs: Additional keyword arguments for LLM configuration (e.g., model_name,
                      api_key, embedding_model_name).

        Returns:
            list[float]: A list of floating-point numbers representing the embedding vector.
        """
        pass

    @abstractmethod
    def get_llm_info(self) -> dict:
        """
        Retrieves information and capabilities of the underlying LLM.

        This can include details like the model name, provider, context window size,
        supported features, etc.

        Returns:
            dict: A dictionary containing information about the LLM.
                  Common keys might include 'model_name', 'provider', 'max_context_length'.
        """
        pass
