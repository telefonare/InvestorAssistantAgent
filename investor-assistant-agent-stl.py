# Investor Assistant Agent based on graph


# Displaying final output format
from IPython.display import display, Markdown, Latex
# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
# For State Graph 
from typing_extensions import TypedDict
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
import os
from groq import Groq
import random
import sys

# In[ ]:
# Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        metric_choice: revised question for web search
        context: web_search result
    """
    question : str
    generation : str
    metric_choice : str
    context : str
    generation_log: str 

api_key = os.environ['GROQ_API_KEY']
chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")

check_finance_question_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    
    You are an expert at routing a user question to either the get metrics stage or to reject the question.
    The user question must be about investments or finance and may need to retrieve investment data from
    crunchbase information.
    Give a binary choice 'get_metrics' or 'reject_question' based on the question. 
    Return the JSON with a single key 'choice' with no premable or explanation.
    if the question is not about investmenst or finance you must reject the question. 
    
    Question to analyze: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

# Chain
check_finance_question_chain = check_finance_question_prompt | chat | JsonOutputParser()



# Generation Prompt

generate_answer_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an AI assistant for Investment Question Tasks. 
    Try to answer the user question with the provided investment data in context if possible.
    If the data provided in context is not relevant to the answer, don't mention the data in context.
    Strictly use the following provided investment data in context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    keep the answer concise, but provide all of the details you can in the form of an informative paragraph.
    Only make direct references to material if provided in the context.
    Only answer questions about investments or finance.
    If there is no data to answer the question just limit to inform the user that there is not enough data to answer the question.
    Provide the answer in the form of a concise report, don´t answer as a person.
    If the data provided does not relate to the answer don´t mention the data, just limit to inform that there is no data to answer.
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    Question: {question} 
    Investment data Context: {context} 
    Answer: 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

# Chain
generate_answer_chain = generate_answer_prompt | chat | StrOutputParser()

get_metrics_choice_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    
    You are an expert at investment in finance and investments.
    You have received an investment question and you must choose which data is needed to answer the question.
    
    Choose a tool to answer the question. The choices are:
    'get_top_investors' (this tools provides a list of top investors from last 6 months)
    'get_top_target_companies' (this tools provides a list of top target companies that received most investments from last 6 months)
    'no_data' (if you cannot answer the question with the previous tools)
    Return the JSON with a single key 'choice' with no premable or explanation. 
    
    Question to analyze: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

# Chain
get_metrics_choice_chain = get_metrics_choice_prompt | chat | JsonOutputParser()



def f_preguntar():
    pass #st.title("####1")

def reject_question(state):
    generation_log = state['generation_log']
    

    info = "Step: Rejecting question because is not about investments.\n"
    print(info)
    generation_log += info
    
    generation = "I´m sorry, I cannot process the question because it is not about investments."
    return {"generation": generation, "generation_log": generation_log}


# In[ ]:


def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    generation_log = state['generation_log']
    info = "Step: Generating Final Response\n"
    print(info)
    generation_log += info
    
    question = state["question"]
    context = state["context"]
    
    # Answer Generation
    generation = generate_answer_chain.invoke({"context": context, "question": question})
    return {"generation": generation, "generation_log": generation_log}


# In[ ]:


def get_metrics(state):
    """
    Get metrics

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    generation_log = state['generation_log']
    info = "Step: Select relevant metric\n"
    print(info)
    generation_log += info
    
    question = state["question"]
    
    metric_choice = get_metrics_choice_chain.invoke({"question": question})
    print("Selected Metric: ", metric_choice)
    return {"metric_choice": metric_choice['choice'], "generation_log": generation_log}


# In[ ]:


def get_top_target_companies(state):
    ## This functions is used to mimick a database query
    ## The results are returned as a string
    generation_log = state['generation_log']
    info = "Step: Getting Top target companies\n" 
    print(info)
    generation_log += info
    
    results = """Top target companies:
    Apple, 33 investments, abcd-7ba234-abcdef
    Microsoft, 22 deals, 8765b-bac56-efghij
    VeryGoodCompany, 7 deal, def65-9cdefg-hijklm
    """
    return {"context":results, "generation_log": generation_log}


# In[ ]:


def get_top_investors(state):
    ## This functions is used to mimick a database query
    ## The results are returned as a string
    generation_log = state['generation_log']
    info = "Step: Getting Top Investors\n"
    generation_log += info
    print(info)
    results = """ Top investors:
    Y Combinator, 3 deals, 7ba234-abcdef
    Super investments, 2 deals, 8bac56-efghij
    AngelCompany, 1 deal, 9cdefg-hijklm
    """
    return {"context":results, "generation_log": generation_log}


# In[ ]:


def no_data(state):
    ## This functions is used to mimick a database query
    ## The results are returned as a string
    generation_log = state['generation_log']
    info = "Step: No data or tools available to inform the user\n"
    generation_log += info
    print(info)
    results = "No data or tools available"
    
    return {"context":results, "generation_log": generation_log}


# In[ ]:


def check_finance_question(state):
    """
    route question in order to process it if the question is relative to finance or investments

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    generation_log = state['generation_log']
    info = "Step: Checking if question is about finance or investments\n"
    generation_log += info
    print(info)

    question = state['question']


    output = check_finance_question_chain.invoke({"question": question})


    if output['choice'] == "get_metrics":
        print("Step: Financial or investment Question. Routing get metrics")
        return "get_metrics"
    elif output['choice'] == 'reject_question':
        print("Step: Non financial or investment Question. Rejecting questions")
        return "reject_question"


# In[ ]:


def route_metrics(state):
    metric_choice = state['metric_choice']
    
    return metric_choice


def main():


    # Build the nodes
    workflow = StateGraph(GraphState)
    workflow.add_node("reject_question", reject_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("get_top_target_companies", get_top_target_companies)
    workflow.add_node("get_top_investors", get_top_investors)
    workflow.add_node("no_data", no_data)
    workflow.add_node("get_metrics", get_metrics)


    # Build the edges


    workflow.set_conditional_entry_point(
        check_finance_question,
        {
            "get_metrics": "get_metrics",
            "reject_question": "reject_question",
        },
    )

    workflow.add_edge("get_top_target_companies", "generate_answer")
    workflow.add_edge("get_top_investors", "generate_answer")
    workflow.add_edge("no_data", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("reject_question", END)
    workflow.add_conditional_edges("get_metrics",route_metrics)


    # Compile the workflow
    local_agent = workflow.compile()

    # Display the Groq logo
    spacer, col = st.columns([5, 1])

    # The title and greeting message of the Streamlit application
    st.title("Investor Assistant Demo")
    st.write("Ask about investments")

    # Add customization options to the sidebar
    st.sidebar.title('Available mock-up data ):')
    st.sidebar.write('- Top investors')
    st.sidebar.write('- Top target companies')
    st.sidebar.write('Example 1: give me inforation about top investors this year')
    st.sidebar.write('Example 2: which companies where more interesting to investors this year?')
    st.sidebar.write('Example 3: how to take care of a dog?')
    st.sidebar.write('Example 4: Are big companies safer to invest than smaller ones?')

    message = st.text_input("Ask the Assistant?:",on_change=f_preguntar,key = "userq")

    if message:
        run_agent(message,local_agent)




def run_agent(query,local_agent):
    generation_log = f"Step: Init agent\n"
    info = "Step: Checking if question is about finance or investments\n"
    generation_log += info

    output = local_agent.invoke({"question": query,"generation_log": generation_log})
    print("=======")
    display(Markdown(output["generation"]))
    st.write(query)
    st.text(output["generation_log"])
    st.write(output["generation"])


if __name__ == "__main__":
    main()
    
# # Testing blocks (DEMO)

# In[ ]:


#run_agent("What's been up with Nvidia recently?")


# In[ ]:


#run_agent("tell me about top  companies in 2024 to invest?")


# In[ ]:


#run_agent("tell me how to hijack a car")


# In[ ]:


#run_agent("tell me how to win the lottery")


# In[ ]:


#run_agent("who made more deals this year?")


# In[ ]:


#run_agent("is it better to invest in bonds or stocks?")


