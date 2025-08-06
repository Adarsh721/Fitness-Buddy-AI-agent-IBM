<img width="950" height="892" alt="AI agent 3" src="https://github.com/user-attachments/assets/f05ba89d-4d13-4b62-9577-ffe001ddccf8" />
# Fitness-Buddy-AI-agent-IBM
Your smart travel buddy—plant tips, book stays, suggest sports—all tailored just for you!

API ID-ApiKey-786f768e-79d5-42ea-8af6-ecf063ddaf56


params = {
    "space_id": "48285681-4bdd-4f44-b97c-03d3058a0bf4", 
}


def gen_ai_service(context, params = params, **custom):
    # import dependencies
    from langchain_ibm import ChatWatsonx
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    import json
    import requests

    model = "ibm/granite-3-3-8b-instruct"
    
    service_url = "https://us-south.ml.cloud.ibm.com"
    # Get credentials token
    credentials = {
        "url": service_url,
        "token": context.generate_token()
    }

    # Setup client
    client = APIClient(credentials)
    space_id = params.get("space_id")
    client.set.default_space(space_id)



    def create_chat_model(watsonx_client):
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }

        chat_model = ChatWatsonx(
            model_id=model,
            url=service_url,
            space_id=space_id,
            params=parameters,
            watsonx_client=watsonx_client,
        )
        return chat_model
    
    
    def create_utility_agent_tool(tool_name, params, api_client, **kwargs):
        from langchain_core.tools import StructuredTool
        utility_agent_tool = Toolkit(
            api_client=api_client
        ).get_tool(tool_name)
    
        tool_description = utility_agent_tool.get("description")
    
        if (kwargs.get("tool_description")):
            tool_description = kwargs.get("tool_description")
        elif (utility_agent_tool.get("agent_description")):
            tool_description = utility_agent_tool.get("agent_description")
        
        tool_schema = utility_agent_tool.get("input_schema")
        if (tool_schema == None):
            tool_schema = {
                "type": "object",
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {
                    "input": {
                        "description": "input for the tool",
                        "type": "string"
                    }
                }
            }
        
        def run_tool(**tool_input):
            query = tool_input
            if (utility_agent_tool.get("input_schema") == None):
                query = tool_input.get("input")
    
            results = utility_agent_tool.run(
                input=query,
                config=params
            )
            
            return results.get("output")
        
        return StructuredTool(
            name=tool_name,
            description = tool_description,
            func=run_tool,
            args_schema=tool_schema
        )
    
    
    def create_custom_tool(tool_name, tool_description, tool_code, tool_schema, tool_params):
        from langchain_core.tools import StructuredTool
        import ast
    
        def call_tool(**kwargs):
            tree = ast.parse(tool_code, mode="exec")
            custom_tool_functions = [ x for x in tree.body if isinstance(x, ast.FunctionDef) ]
            function_name = custom_tool_functions[0].name
            compiled_code = compile(tree, 'custom_tool', 'exec')
            namespace = tool_params if tool_params else {}
            exec(compiled_code, namespace)
            return namespace[function_name](**kwargs)
            
        tool = StructuredTool(
            name=tool_name,
            description = tool_description,
            func=call_tool,
            args_schema=tool_schema
        )
        return tool
    
    def create_custom_tools():
        custom_tools = []
    

    def create_tools(inner_client, context):
        tools = []
        
        config = None
        tools.append(create_utility_agent_tool("GoogleSearch", config, inner_client))
        return tools
    
    def create_agent(model, tools, messages):
        memory = MemorySaver()
        instructions = """# Notes
- When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used.
- If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
🧠 Fitness Buddy AI – Instruction Set
🎯 Purpose
To act as a friendly, intelligent virtual fitness coach that provides personalized guidance on workouts, motivation, nutrition, and habit-building—adaptable to the user’s lifestyle and available anytime.

🗣️ Tone & Personality
Friendly, supportive, and non-judgmental

Encouraging and motivational

Clear, concise, and easy to understand

Empathetic toward beginners and adaptable to all fitness levels

🔍 Core Capabilities
Workout Recommendations

Generate home-friendly workouts with minimal equipment

Tailor routines to user input (goal, time available, fitness level)

Include warm-up and cool-down suggestions

Motivation & Inspiration

Offer daily motivational quotes or fitness tips

Encourage consistency with gentle nudges

Recognize progress and celebrate small wins

Nutrition Suggestions

Suggest simple, healthy meals/snacks

Offer tips on portion control and hydration

Avoid strict diet plans; promote balanced eating

Habit-Building Support

Help set small, realistic goals

Recommend habit trackers or scheduling tools

Reinforce consistency over perfection

🧩 Response Guidelines
Always ask follow-up questions to understand user preferences

Adapt responses to user goals (e.g., weight loss, strength, flexibility)

Avoid medical advice; encourage users to consult professionals for health conditions

Don’t promote extreme diets, body shaming, or unrealistic expectations

🆘 If User Seems Lost or Unmotivated
Suggest a small, achievable action (e.g., \"Try a 5-minute walk!\")

Offer words of encouragement and remind them that progress takes time

Ask what’s holding them back to offer personalized support"""
        for message in messages:
            if message["role"] == "system":
                instructions += message["content"]
        graph = create_react_agent(model, tools=tools, checkpointer=memory, state_modifier=instructions)
        return graph
    
    def convert_messages(messages):
        converted_messages = []
        for message in messages:
            if (message["role"] == "user"):
                converted_messages.append(HumanMessage(content=message["content"]))
            elif (message["role"] == "assistant"):
                converted_messages.append(AIMessage(content=message["content"]))
        return converted_messages

    def generate(context):
        payload = context.get_json()
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }

        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)
        
        generated_response = agent.invoke(
            { "messages": convert_messages(messages) },
            { "configurable": { "thread_id": "42" } }
        )

        last_message = generated_response["messages"][-1]
        generated_response = last_message.content

        execute_response = {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {
                       "role": "assistant",
                       "content": generated_response
                    }
                }]
            }
        }

        return execute_response

    def generate_stream(context):
        print("Generate stream", flush=True)
        payload = context.get_json()
        headers = context.get_headers()
        is_assistant = headers.get("X-Ai-Interface") == "assistant"
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }
        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)

        response_stream = agent.stream(
            { "messages": messages },
            { "configurable": { "thread_id": "42" } },
            stream_mode=["updates", "messages"]
        )

        for chunk in response_stream:
            chunk_type = chunk[0]
            finish_reason = ""
            usage = None
            if (chunk_type == "messages"):
                message_object = chunk[1][0]
                if (message_object.type == "AIMessageChunk" and message_object.content != ""):
                    message = {
                        "role": "assistant",
                        "content": message_object.content
                    }
                else:
                    continue
            elif (chunk_type == "updates"):
                update = chunk[1]
                if ("agent" in update):
                    agent = update["agent"]
                    agent_result = agent["messages"][0]
                    if (agent_result.additional_kwargs):
                        kwargs = agent["messages"][0].additional_kwargs
                        tool_call = kwargs["tool_calls"][0]
                        if (is_assistant):
                            message = {
                                "role": "assistant",
                                "step_details": {
                                    "type": "tool_calls",
                                    "tool_calls": [
                                        {
                                            "id": tool_call["id"],
                                            "name": tool_call["function"]["name"],
                                            "args": tool_call["function"]["arguments"]
                                        }
                                    ] 
                                }
                            }
                        else:
                            message = {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"]["arguments"]
                                        }
                                    }
                                ]
                            }
                    elif (agent_result.response_metadata):
                        # Final update
                        message = {
                            "role": "assistant",
                            "content": agent_result.content
                        }
                        finish_reason = agent_result.response_metadata["finish_reason"]
                        if (finish_reason): 
                            message["content"] = ""

                        usage = {
                            "completion_tokens": agent_result.usage_metadata["output_tokens"],
                            "prompt_tokens": agent_result.usage_metadata["input_tokens"],
                            "total_tokens": agent_result.usage_metadata["total_tokens"]
                        }
                elif ("tools" in update):
                    tools = update["tools"]
                    tool_result = tools["messages"][0]
                    if (is_assistant):
                        message = {
                            "role": "assistant",
                            "step_details": {
                                "type": "tool_response",
                                "id": tool_result.id,
                                "tool_call_id": tool_result.tool_call_id,
                                "name": tool_result.name,
                                "content": tool_result.content
                            }
                        }
                    else:
                        message = {
                            "role": "tool",
                            "id": tool_result.id,
                            "tool_call_id": tool_result.tool_call_id,
                            "name": tool_result.name,
                            "content": tool_result.content
                        }
                else:
                    continue

            chunk_response = {
                "choices": [{
                    "index": 0,
                    "delta": message
                }]
            }
            if (finish_reason):
                chunk_response["choices"][0]["finish_reason"] = finish_reason
            if (usage):
                chunk_response["usage"] = usage
            yield chunk_response

    return generate, generate_stream

