import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from openai import OpenAI, APIConnectionError, APITimeoutError
from groq import Groq
import ollama
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np


_EMPTY_CHOICES_RETRIES = 3
_EMPTY_CHOICES_BACKOFF = 0.5  # seconds; doubles each attempt
_REQUEST_TIMEOUT_SECONDS = 60.0  # per-request read timeout for OpenAI/OpenRouter calls


def _extract_provider_error(chat_completion) -> str:
    err = getattr(chat_completion, "error", None)
    if err is None:
        extra = getattr(chat_completion, "model_extra", None) or {}
        err = extra.get("error") if isinstance(extra, dict) else None
    if isinstance(err, dict):
        return err.get("message") or str(err)
    if err is not None:
        return str(err)
    return "no choices and no error field"


def _empty_choice_stub(detail: str):
    return SimpleNamespace(
        message=SimpleNamespace(content=None, refusal=None),
        finish_reason=f"empty_choices: {detail}" if detail else "empty_choices",
    )

class SelfCheckAPIPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via API-based prompting (e.g., OpenAI's GPT)
    """
    def __init__(
        self,
        client_type = "openai",
        base_url = "https://ollama.makinteract.com/v1/",
        model = "gpt-3.5-turbo",
        api_key = None,
    ):
        if client_type == "openai":
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            print("Initiate OpenAI client... model = {}".format(model))
        if client_type == "ollama":
            self.client = ollama.Client(host=base_url)
            print("Initiate Ollama client... model = {}".format(model))
        if client_type == "groq":
            self.client = Groq(api_key=api_key)
            print("Initiate Groq client... model = {}".format(model))
        
        self.client_type = client_type
        self.model = model
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()


    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def completion(self, prompt: str, max_tokens: int = 10000, reasoning="none"):
        if self.client_type in ("openai", "groq"):
            last_detail = ""
            for attempt in range(_EMPTY_CHOICES_RETRIES):
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning,
                    )
                except (APITimeoutError, APIConnectionError) as exc:
                    last_detail = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[Warning] request failed "
                        f"(attempt {attempt + 1}/{_EMPTY_CHOICES_RETRIES}): {last_detail}"
                    )
                else:
                    if chat_completion.choices:
                        return chat_completion.choices[0]
                    last_detail = _extract_provider_error(chat_completion)
                    print(
                        f"[Warning] provider returned no choices "
                        f"(attempt {attempt + 1}/{_EMPTY_CHOICES_RETRIES}): {last_detail}"
                    )
                if attempt + 1 < _EMPTY_CHOICES_RETRIES:
                    time.sleep(_EMPTY_CHOICES_BACKOFF * (2 ** attempt))
            return _empty_choice_stub(last_detail)

        elif self.client_type == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": max_tokens},
                think=reasoning != "none",
            )
            return response

        else:
            raise ValueError("client_type not implemented")

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        max_tokens: int = 10000,
        reasoning="none",
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param max_tokens: int -- maximum number of tokens to generate
        :param reasoning: str -- reasoning effort level
        :param verbose: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        responses = []
        disable = not verbose

        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            responses.append([])

            def _call(sample_i: int, sample: str) -> tuple[int, str | None]:
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_text = self.completion(prompt, max_tokens=max_tokens, reasoning=reasoning)
                message = generate_text.message.content
                if message is None:
                    finish_reason = getattr(generate_text, 'finish_reason', None) or getattr(generate_text, 'done_reason', None)
                    refusal = getattr(generate_text.message, 'refusal', None)
                    print(f"[Warning] API None for {prompt}\n -> reason: {finish_reason}, refusal: {refusal}")
                return sample_i, message

            with ThreadPoolExecutor(max_workers=num_samples+10) as executor:
                futures = {executor.submit(_call, i, s): i for i, s in enumerate(sampled_passages)}
                sample_results = [None] * num_samples
                for future in as_completed(futures):
                    sample_i, message = future.result()
                    sample_results[sample_i] = message
                    score = self.text_postprocessing(message)
                    scores[sent_i, sample_i] = score
                    print(f"  sent {sent_i+1}/{num_sentences} sample {sample_i+1}/{num_samples} → {message!r} (score={score:.1f})")

            responses[sent_i].extend(sample_results)
        
        scores = scores.mean(axis=-1)
        return scores, responses

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        if text is None:
            return self.text_mapping['n/a']
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
