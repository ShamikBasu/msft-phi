
from langchain import LLMChain, PromptTemplate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate
torch.random.manual_seed(0)

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
    max_new_tokens= 900,
    return_full_text = False,
    temperature  = 0.1,
    do_sample = False,
    tokenizer=tokenizer,
)

# Wrap the pipeline in a Langchain LLM wrapper
slm = HuggingFacePipeline(pipeline=pipe)

def create_prompt():
    #System Prompt Template
    system_template = "You are an AI Assistant that specializes in association football concepts, make it brief, interesting and concise."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Human Prompt Template
    human_template = "{football_query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    #Chat Prompt Template
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt, human_message_prompt
    ])
    return  chat_prompt


chat_prompt = create_prompt()
chat_chain = LLMChain(llm = slm , prompt = chat_prompt)

user_query = input("What is your question?")
response  = chat_chain.run(football_query = user_query)
print(f"AI Assistant: {response}")
