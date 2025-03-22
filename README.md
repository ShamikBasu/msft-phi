# [ msft-phi - Microsoft's Phi Open Models Small Language Models](https://azure.microsoft.com/en-us/products/phi)
Proof of Concept for MSFT PHI SLMs - [ **Microsoft Phi 3.5** ](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/4225280)


SLM -> [**Small Language Models**](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-small-language-models)

Implementation -> Uses [**microsoft/Phi-3.5-mini-instruct**](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) integrated with [**Langchain**](https://python.langchain.com/docs/introduction/)

To check more Phi Models [click here](https://ai.azure.com/explore/models?selectedCollection=phi) 

Integrations:
1. [Unstructured Email Loader with Pydantic Outpur Parser](langchain-document-loader/phi-langchain-integrations.py)
2. [Runnable Message History Chain](langchain-runnables/MessageHistoryChain.py)
3. [Few Shot Prompting](prompt-template/phi-few-shot-prompt-template.py)
4. [Sequential Chain](langchanin-chains/phi-langchain-sequential-chain.py)
5. [Conversation Buffer Memory](langchain-memory/phi-langchain-conversation-buffer-memory.py)

***Note - This is HuggingFace Implementation, refer to [Azure AI Inference](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme?view=azure-python-preview) code samples when deploying models to Azure AI***

