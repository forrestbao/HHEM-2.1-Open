
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from IPython.display import Markdown

# Load the models
hhem_2_model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)

hhem_1_model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True, revision='hhem-1.0-open')

hhem_1_tokenizer = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model', revision='hhem-1.0-open')


def __hhem1_predict(premise: str, hypothesis: str) -> float:
    model = hhem_1_model
    tokenizer = hhem_1_tokenizer

    inputs = tokenizer.batch_encode_plus([(premise, hypothesis)], return_tensors='pt', padding=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().detach().numpy()
        # convert logits to probabilities
        scores = 1 / (1 + np.exp(-logits)).flatten()

    return scores[0]

def __hhem2_predict(
        premise: str, 
        hypothesis: str
        ) -> float:
    
    model = hhem_2_model

    pairs = [(premise, hypothesis)]    
    scores = model.predict(pairs)   
    return scores[0]


def HHEM_web(
        premise: str = "Vectara's HHEM-2.1-Open is the best open-source hallucination evaluation model outperforming GPT-4 and runnable on consumer-level GPUs like the RTX 3080.", 
        hypothesis: str = "Vectara's open-source HHEM model beat GPT-4 and you can run it on a consumer-grade hardware. Vectara builds hallucination detection and RAG models. "
        ) -> Markdown:
    """ # The web demo of [Vectara](https://vectara.com)'s [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model)

    **Disclaimer**: This app is personally maintained by Vectara lovers on [Github](https://github.com/forrestbao/HHEM-web-app). Opinions expressed here do not reflect those of Vectara. 
    
    **Usage**: Simply enter a premise and a hypothesis and click the Run button. HHEM will give you a score, between 0 (most hallucinated) and 1 (most consistent), gauging how well the hypothesis is supported by the premise. 

    For example, the placeholder example (if you don't see it, refresh your web browser) below is hallucinated. If you remove "and RAG" and re-run, the score from HHEM-2.1-Open will jump from 0.36 to 0.90.

    """

    if len(premise) + len(hypothesis) > 2000*4: 
        return "To avoid out-of-memory issue, this app requires premise and hypothesis to be under 8000 characters (which is about 2000 tokens in English) altogether." 

    hhem_1_score = __hhem1_predict(premise, hypothesis)
    hhem_2_score = __hhem2_predict(premise, hypothesis)

    return Markdown(f"""
Factual consistency scores by two versions of HHEM: 
                    
|1.0-Open|2.0-Open| 
|----|----|
| {hhem_1_score:.3f} | {hhem_2_score:.3f}|
    """)