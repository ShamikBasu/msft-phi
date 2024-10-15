import torch
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain

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
    max_new_tokens= 2000,
    return_full_text = False,
    temperature  = 0,
    top_p = 0.2,
    do_sample = False,
    tokenizer=tokenizer,
)

slm = HuggingFacePipeline(pipeline = pipe)

first_template  = """
        You are an doctor assistant, who occasionally helps patients with their basic queries.
        This is the issue the patient is facing {issue} as symptoms.
        Write 5 pointed causes for this issue and symptom.
"""
first_prompt = ChatPromptTemplate.from_template(first_template)
first_chain = LLMChain(llm= slm, prompt =first_prompt, output_key = "issue_points")

second_template  = """
        You are an doctor assistant, who occasionally helps patients with their basic queries.
        Based on the following issue points, give precautions and the care needed. 
        {issue_points} 
        Be precise and summarise it.
        Don't provide huge response
"""
second_prompt = ChatPromptTemplate.from_template(second_template)
second_chain = LLMChain(llm= slm, prompt =second_prompt, output_key = "care")

third_template  = """
        You are an doctor assistant, who occasionally helps patients with their basic queries.
        Based on the following care 
        {care}
        devise a very caring and good routine in a concise way for the patient and in the end say
"""
third_prompt = ChatPromptTemplate.from_template(third_template)
third_chain = LLMChain(llm= slm, prompt =third_prompt, output_key = "final_plan")


sequential_chain = SequentialChain(
            chains = [first_chain, second_chain, third_chain],
            input_variables = ["issue"],
            output_variables = ["issue_points","care", "final_plan"],
            verbose = True
)

result = sequential_chain(input('WHAT PROBLEM YOU ARE FACING?'))

print("AI RESPONSE::", result)
print("AI KEYS::", result.keys())
print("ISSUE :", result['issue'])
print("ISSUE POINTS :", result['issue_points'])
print("CARE :", result['care'])
print("FINAL PLAN :", result['final_plan'])


