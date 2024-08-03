
from transformers import AutoModelForSequenceClassification
from IPython.display import Markdown

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)

def HHEM2_Web(
        premise: str = "Vectara's HHEM-2.1-Open is the best open-source hallucination evaluation model outperforming GPT-4 and runnable on consumer-level GPUs like the RTX 3080.", 
        hypothesis: str = "Vectara's open-source HHEM model beat GPT-4 and you can run it on a consumer-grade hardware. Vectara builds hallucination detection and RAG models. "
        ) -> Markdown:
    """ # The web demo of [Vectara](https://vectara.com)'s [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model)

    ### Disclaimer

    This app is personally maintained by Vectara lovers on [Github](https://github.com/forrestbao/HHEM-2.1-Open). Opinions expressed here do not reflect those of Vectara. 

    
    ### Usage 

    Simply enter a premise and a hypothesis and [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model) will give you a score, between 0 (very hallucinated) and 1 (very consistent), gauging how well the hypothesis is supported by the premise. 

    The placeholder example below is hallucinated. Can you tell which part is hallucinated? Hint: Try to remove "and RAG" and see if the score changes.

    """
    pairs = [(premise, hypothesis)]
    if len(premise) + len(hypothesis) > 2000*4: 
        return "Your input is too long. Please ensure that premise and hypothesis is under 2000 characters altogether." 
    scores = model.predict(pairs)   
    return f"The score is {scores[0]:.2f}."
