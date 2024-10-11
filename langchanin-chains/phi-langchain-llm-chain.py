from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.random.manual_seed(0)

from langchain import HuggingFacePipeline
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain import  LLMChain

import warnings
warnings.filterwarnings('ignore')
# SLM Creation
# Declare the model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    max_new_tokens= 1000,
    return_full_text = False,
    temperature  = 0.2,
    do_sample = False,
    tokenizer=tokenizer,
)

slm = HuggingFacePipeline(pipeline = pipe)

#Prompt Template
human_prompt = HumanMessagePromptTemplate.from_template("Can you provide recipe for making {food_item}")
prompt = ChatPromptTemplate.from_messages([human_prompt])

#create chain -> with slm, prompt
chain = LLMChain(llm = slm , prompt = prompt, verbose = True)
result = chain.run(food_item = input("What do you want to cook today?"))
print("AI RESPONSE \n", result)