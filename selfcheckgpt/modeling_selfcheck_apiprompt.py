from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from groq import Groq
import ollama
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np

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
            self.client = OpenAI(base_url=base_url, api_key=api_key)
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
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=reasoning,
            )
            return chat_completion.choices[0]

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
