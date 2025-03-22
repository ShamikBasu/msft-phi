from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.random.manual_seed(0)

from langchain import HuggingFacePipeline

import warnings

warnings.filterwarnings("ignore")
#SLM
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
    max_new_tokens= 500,
    return_full_text = False,
    temperature  = 0,
    top_p = 0.2,
    do_sample = False,
    tokenizer=tokenizer,
)

slm = HuggingFacePipeline(pipeline = pipe)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm = model, memory = memory)
result = conversation.predict("HELLO")
print("MEMORY:: ", memory)
print("MEMORY MESSAGES::: ", memory.buffer)
print('AI RESPONSE:::', result)