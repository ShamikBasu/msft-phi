import torch
from langchain.chains.sequential import SimpleSequentialChain

torch.random.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import LLMChain, HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
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
    max_new_tokens= 700,
    return_full_text = False,
    temperature  = 0.2,
    do_sample = False,
    tokenizer=tokenizer,
)

slm = HuggingFacePipeline(pipeline = pipe)

first_template = '''
    You are an advanced ai assistant who generates 10 bullet points for the below topic 
    {topic}
'''
first_prompt = ChatPromptTemplate.from_template(first_template)
first_chain = LLMChain(llm = slm, prompt = first_prompt)

second_template =  '''
    You are an advanced ai article generator, which generates the article for 
    {outline}
    
    The article should be easily comprehensible yet very elegant for a prestigious magazine.
'''
second_prompt = ChatPromptTemplate.from_template(second_template)
second_chain = LLMChain(llm = slm, prompt = second_prompt)


# Full chain
complete_simple_chain = SimpleSequentialChain(
                        chains = [ first_chain, second_chain],
                        verbose = True
)

result = complete_simple_chain.run(input("What is your topic?"))
print("AI RESPONSE \n", result)