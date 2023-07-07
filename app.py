import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline(
            'text-generation',
            model='tiiuae/falcon-7b',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=0,
        )

    def infer(self, prompt):
        pipeline_output = self.generator(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1
        )
        generated_text = pipeline_output[0]['generated_text']
        return generated_text

    def finalize(self):
        self.pipe = None