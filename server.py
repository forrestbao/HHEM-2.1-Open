# The code is taken verbatim from https://huggingface.co/vectara/hallucination_evaluation_model

from transformers import AutoModelForSequenceClassification
from IPython.display import Markdown

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)

def HalluOMeter(
        premise: str = "Vectara's HHEM-2.1-Open is the best open-source hallucination evaluation model that you can run on consumer-level GPUs like the RTX 3080.", 
        hypothesis: str = "Vectara is a company that makes the best hallucination evaluation models."
        ) -> Markdown:
    """ # The web demo of [Vectara](https://vectara.com)'s HHEM-2.1-Open

    Simply enter a premise and a hypothesis and HHEM-2.1-Open will give you a score, between 0 and 1, gauging how well the hypothesis is supported by the premise. The higher the score, the less the hypothesis hallucinates. For more information, see [here](https://huggingface.co/vectara/hallucination_evaluation_model). 

    """
    pairs = [(premise, hypothesis)]
    try: 
        scores = model.predict(pairs)
    except: 
        return "Error: Your input may be too long. Please try again with shorter (e.g., <1k tokens) inputs."
    
    return f"The hallucination score is {scores[0]:.2f}."