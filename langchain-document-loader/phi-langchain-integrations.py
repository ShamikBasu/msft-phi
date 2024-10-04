from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.random.manual_seed(0)
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from  pydantic import BaseModel,Field

#Email Loader
email_loader = UnstructuredEmailLoader("test.eml")
data = email_loader.load()

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
    max_new_tokens= 900,
    return_full_text = False,
    temperature  = 0.2,
    do_sample = False,
    tokenizer=tokenizer,
)

class EmailClassification(BaseModel):
    summary : str = Field(description= "Summary of the email")
    category : str = Field(description= "Category of the email")
    sender : str = Field(description= "Who sent the email")

output_parser = PydanticOutputParser(pydantic_object = EmailClassification)
#print(output_parser.get_format_instructions())
#print("Email data", data[0].page_content)

# Create Prompt
template = """
        You are an Email Classifier tool. 
        Your job is to 
         1. summarize the email
         2. provide the category of the email
         3. provide who sent the email
        {email}\n{format_instructions}
        """

prompt = PromptTemplate.from_template(template).format(email = data[0].page_content , format_instructions = output_parser.get_format_instructions())
slm = HuggingFacePipeline(pipeline = pipe)
result = slm.invoke(prompt)
print("AI RESPONSE ::",result)
parsed_result = output_parser.parse(result)
print("AI RESPONSE PARSED::", parsed_result)
summary = parsed_result.summary
category = parsed_result.category
sender = parsed_result.sender
print("SUMMARY OF EMAIL:", summary)
print("CATEGORY OF EMAIL:", category)
print("SENDER OF EMAIL:", sender)







