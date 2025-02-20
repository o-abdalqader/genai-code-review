"""
Este módulo contém a classe AzureOpenAIClient, que é usada para interagir com a API do Azure OpenAI.
A classe AzureOpenAIClient pode ser usada para gerar respostas de um modelo especificado do Azure OpenAI.
"""

import logging
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AzureOpenAIClient:
    """
    A client for interacting with the Azure OpenAI API to generate responses using a specified model.
    """

    def __init__(self, model, temperature, max_tokens, api_version, azure_endpoint):
        """
        Initialize the AzureOpenAIClient with model, temperature, max tokens, api_version, and azure_endpoint.

        Args:
            model (str): The Azure OpenAI model to use.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
            api_version (str): The API version to use.
            azure_endpoint (str): The Azure OpenAI endpoint.
        """
        try:
            self.client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint)
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.api_version = api_version
            self.azure_endpoint = azure_endpoint
            logging.info(
                "Azure OpenAI client initialized successfully, "
                "Model: %s, temperature: %s, max tokens: %s, API version: %s, Azure endpoint: %s",
                self.model,
                self.temperature,
                self.max_tokens,
                self.api_version,
                self.azure_endpoint
            )
        except Exception as e:
            logging.error("Error initializing Azure OpenAI client: %s", e)
            raise

    def generate_response(self, prompt):
        """
        Generate a response from the Azure OpenAI model based on the given prompt.

        Args:
            prompt (str): The prompt to send to the Azure OpenAI API.

        Returns:
            str: The generated response from the Azure OpenAI model.

        Raises:
            Exception: If there is an error generating the response.
        """
        try:
            logging.info("Generating response from Azure OpenAI model.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint
            )
            logging.info("Response generated successfully.")
            return response.choices[0].message.content
        except Exception as e:
            logging.error("Error generating response from Azure OpenAI model: %s", e)
            raise
