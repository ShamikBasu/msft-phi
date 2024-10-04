from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import HumanMessagePromptTemplate,AIMessagePromptTemplate,SystemMessagePromptTemplate,ChatPromptTemplate
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
    temperature  = 0.5,
    do_sample = False,
    tokenizer=tokenizer,
)



def create_few_shot_prompt_template():
    system_template = 'You are a an AI Assistant that generates one liner headlines from news article passed to you.'
    system_prompt = HumanMessagePromptTemplate.from_template(system_template)

    example_input_one = " Lionel Messi's arrival at Inter Miami has sparked a major resurgence for the MLS club. His goals and assists have propelled the team to the top of the Eastern Conference, attracting new fans and generating significant revenue. Messi's presence has also elevated the league's overall profile, drawing international attention to Major League Soccer."
    example_output_one = "Messi's Impact on Inter Miami"
    example_input_prompt_one = HumanMessagePromptTemplate.from_template(example_input_one)
    example_output_prompt_one = AIMessagePromptTemplate.from_template(example_output_one)

    example_input_two = "Women's football is experiencing unprecedented growth worldwide. The FIFA Women's World Cup, held earlier this year, set new viewership records and showcased the talent and passion of female players. The tournament's success has led to increased investment in women's football leagues and facilities, as well as greater recognition for female athletes."
    example_output_two = "Women's Football Continues to Grow"
    example_input_prompt_two = HumanMessagePromptTemplate.from_template(example_input_two)
    example_output_prompt_two = AIMessagePromptTemplate.from_template(example_output_two)

    human_template = "{news_article}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_prompt, example_input_prompt_one,example_output_prompt_one,example_input_prompt_two, example_output_prompt_two, human_message_prompt
    ])

    return chat_prompt

chat_prompt = create_few_shot_prompt_template()
slm = HuggingFacePipeline(pipeline = pipe)
llm_chain = LLMChain(llm = slm, prompt = chat_prompt )
result = llm_chain.run(news_article = input("Please paste your news article"))
print(f"AI Assistant: {result}")