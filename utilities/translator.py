"""
Translate text using a pre-trained sequence-to-sequence language model.
https://huggingface.co/facebook/nllb-200-distilled-1.3B
Author: Gary A. Stafford
Date: 2025-04-15
"""

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Translator:
    def __init__(self):
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                ),
            )
            .to(DEVICE)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def translate_text(self, text, language="eng_Latn") -> str:
        """
        Translates the given text to the specified language.
        Args:
            text (str): The text to be translated.
            language (str): The language code to translate the text into.
                Default is "eng_Latn" (English in Latin script).
        Returns:
            str: The translated text.
        """

        logging.info(f"Translating text to: {language}...")

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(language),
            max_length=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
        response = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]

        return response
