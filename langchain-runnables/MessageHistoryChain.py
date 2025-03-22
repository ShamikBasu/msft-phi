from typing import Optional
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.random.manual_seed(0)
import uuid
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from Memory import get_by_session_id

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
    temperature  = 0.8,
    do_sample = False,
    tokenizer=tokenizer,
)

slm = HuggingFacePipeline(pipeline = pipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{topic}"),
])

chain = prompt | slm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

session_id = str(uuid.uuid4())

print("AI ANSWER: ",chain_with_history.invoke(  # noqa: T201
    {"ability": "teaching", "topic": "High School Physics"},
    config={"configurable": {"session_id": session_id}}
))

store = get_by_session_id(session_id)

print("store", store)
