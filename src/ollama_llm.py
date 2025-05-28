import ollama
from typing import AsyncIterator
from src.llm_interface import BaseLLMInterface

class OllamaLLM(BaseLLMInterface):
    """
    An implementation of BaseLLMInterface for interacting with Ollama models.
    """

    def __init__(self, model_name: str = "moondream", ollama_client: ollama.AsyncClient = None):
        """
        Initializes the OllamaLLM client.

        Args:
            model_name (str): The name of the Ollama model to use (e.g., "moondream", "llama2").
                              Defaults to "moondream".
            ollama_client (ollama.AsyncClient, optional): An existing Ollama async client.
                                                         If None, a new client is created.
        """
        self.model_name = model_name
        self.ollama_client = ollama_client if ollama_client else ollama.AsyncClient()

    async def stream_chat(self, prompt: str, history: list = None, **kwargs) -> AsyncIterator[str]:
        """
        Streams chat responses from the Ollama model.

        This method sends a prompt (and optional conversation history) to the Ollama model
        and yields response content chunks as they become available.

        The `model_name` used for this call can be overridden by passing `model_name="your-chat-model"`
        in `**kwargs`. Other Ollama-specific options (like `temperature`, `top_p`, etc.) can also be
        passed via `**kwargs`.

        Args:
            prompt (str): The user's input/prompt to the LLM.
            history (list, optional): A list of previous conversation turns. Each turn should be a
                                      dictionary with "role" (e.g., "user", "assistant") and "content".
                                      Example: `[{"role": "user", "content": "Hello"},
                                                {"role": "assistant", "content": "Hi there!"}]`
                                      Defaults to None, meaning no history is sent.
            **kwargs: Additional keyword arguments to pass to the Ollama client's chat method.
                      This can include `model_name` (or `model`) to override `self.model_name`,
                      `temperature`, `options`, etc.

        Yields:
            str: Chunks of the LLM's response content.

        Raises:
            ollama.ResponseError: For API errors returned by Ollama.
            Exception: For other unexpected errors during the process.
        """
        if not self.ollama_client:
            # This ensures ollama_client is available, though __init__ should always set it.
            self.ollama_client = ollama.AsyncClient()

        # Determine the model to use
        model_to_use = kwargs.pop("model", None) or kwargs.pop("model_name", self.model_name)

        # Prepare messages payload
        messages = []
        if history:
            messages.extend(history) # Ensure it's a list of dicts as per Ollama format
        messages.append({"role": "user", "content": prompt})

        try:
            stream_response = await self.ollama_client.chat(
                model=model_to_use,
                messages=messages,
                stream=True,
                **kwargs # Pass other options like temperature, top_p, etc.
            )

            async for chunk in stream_response:
                if "message" in chunk and "content" in chunk["message"]:
                    content_piece = chunk["message"]["content"]
                    if content_piece: # Ensure we don't yield empty strings if the API sends them
                        yield content_piece
                # Handle potential 'done' status if necessary, though typically just iterating
                # until the stream ends is sufficient. If 'done' is true and there's an error message,
                # it might be worth logging or raising.
                if chunk.get("done") and chunk.get("error"):
                    raise ollama.ResponseError(f"Ollama stream ended with error: {chunk.get('error')}")

        except ollama.ResponseError as e:
            print(f"Ollama API error during streaming chat for model '{model_to_use}': {e.status_code} - {e.error}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred during streaming chat: {e}")
            raise e
        # No explicit `if False: yield` needed as the `async for` makes it an async generator.

    async def get_embedding(self, text: str, **kwargs) -> list[float]:
        """
        Generates an embedding (vector representation) for the given text using an Ollama model.

        This method calls the Ollama `/api/embeddings` endpoint.
        Ensure that `self.model_name` is set to a model specifically trained or suitable
        for text embeddings (e.g., "nomic-embed-text", "mxbai-embed-large", "all-minilm").
        Using a general-purpose chat model (like "llama2", "moondream") for embeddings might
        produce suboptimal or incorrect embedding vectors, or may not be supported by the model.

        The `model_name` used for this call can be overridden by passing `model_name="your-embedding-model"`
        in `**kwargs`, but it's generally recommended to initialize the OllamaLLM instance with an
        appropriate model if its primary purpose is embedding, or to set `self.model_name` before calls.

        Args:
            text (str): The input text to embed.
            **kwargs: Additional keyword arguments to pass to the Ollama client's embeddings method.
                      This can include `model_name` to override `self.model_name` for this call,
                      or other options supported by the Ollama API for embeddings.

        Returns:
            list[float]: A list of floating-point numbers representing the embedding vector.
        
        Raises:
            KeyError: If the 'embedding' key is not found in the response from Ollama.
            ollama.ResponseError: For API errors returned by Ollama.
            Exception: For other unexpected errors during the process.
        """
        if not self.ollama_client:
            # This ensures ollama_client is available, though __init__ should always set it.
            self.ollama_client = ollama.AsyncClient()

        # Use model_name from kwargs if provided, otherwise default to self.model_name
        model_to_use = kwargs.pop("model", None) or kwargs.pop("model_name", self.model_name)

        try:
            response = await self.ollama_client.embeddings(
                model=model_to_use,
                prompt=text,
                **kwargs  # Pass any other relevant kwargs (e.g., options)
            )
        except ollama.ResponseError as e:
            # Catching specific ollama errors can be useful for tailored error handling
            print(f"Ollama API error while generating embeddings for model '{model_to_use}': {e.status_code} - {e.error}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred while generating embeddings: {e}")
            raise e
        
        if "embedding" not in response:
            # This case should ideally be rare if the API call itself was successful
            # and the model is an embedding model.
            raise KeyError(
                f"The 'embedding' key was not found in the response from Ollama for model '{model_to_use}'. "
                f"Response: {response}"
            )
        
        return response["embedding"]

    def get_llm_info(self) -> dict:
        """
        Retrieves information about the Ollama LLM.

        Returns:
            dict: A dictionary containing information like model name and provider.
        """
        return {
            "model_name": self.model_name,
            "provider": "ollama",
            # Potentially add more info here in the future, e.g., by calling self.ollama_client.list()
            # and finding details for the specific self.model_name, or client/host info.
        }

    async def summarize_image(self, image_path: str, user_prompt: str = "Summarize the contents of this image.") -> dict:
        """
        Generates a summary for the given image using the Ollama model.

        This method adapts the core logic from the previous `summarize_image_document` function.

        Args:
            image_path (str): The path to the image file.
            user_prompt (str, optional): The prompt to use for summarization.
                                      Defaults to "Summarize the contents of this image.".

        Returns:
            dict: A dictionary containing the image_path and its summary.
                  Example: {"file_path": "path/to/image.jpg", "summary": "A description of the image."}
        """
        if not self.ollama_client:
            # This check is more for robustness, as __init__ should always set it.
            self.ollama_client = ollama.AsyncClient()

        chat_completion = await self.ollama_client.chat(
            model=self.model_name, # Uses the model_name specified during class initialization
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_path],
                }
            ],
            # format="json", # Not universally supported, and we're constructing the JSON ourselves
            options={"num_predict": 128}, # As used in the original function
        )

        summary = {
            "file_path": image_path,
            "summary": chat_completion["message"]["content"],
        }
        # The print statements from the original function are removed as they are side effects
        # not suitable for a class method that should return data.
        return summary

    def summarize_image_sync(self, image_path: str, user_prompt: str = "Summarize the contents of this image.") -> dict:
        """
        Generates a summary for the given image using the Ollama model (synchronous version).

        Args:
            image_path (str): The path to the image file.
            user_prompt (str, optional): The prompt to use for summarization.
                                      Defaults to "Summarize the contents of this image.".

        Returns:
            dict: A dictionary containing the image_path and its summary.
                  Example: {"file_path": "path/to/image.jpg", "summary": "A description of the image."}
        """
        # Instantiate a synchronous Ollama client for this method
        sync_ollama_client = ollama.Client()

        chat_completion = sync_ollama_client.chat(
            model=self.model_name, # Uses the model_name specified during class initialization
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_path],
                }
            ],
            options={"num_predict": 128}, # As used in the original function
        )

        summary = {
            "file_path": image_path,
            "summary": chat_completion["message"]["content"],
        }
        return summary
