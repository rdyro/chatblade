import collections
import yaml
import traceback
import warnings

import tiktoken

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.genai as genai
    from google.genai import types

from . import utils, errors

MODEL_ROLE = "model"  # or 'assistant' in openai


class Message(collections.namedtuple("Message", ["role", "content"])):
    @staticmethod
    def represent_for_yaml(dumper, msg):
        val = []
        md = msg._asdict()

        for fie in msg._fields:
            val.append([dumper.represent_data(e) for e in (fie, md[fie])])

        return yaml.nodes.MappingNode("tag:yaml.org,2002:map", val)

    @classmethod
    def import_yaml(cls, seq):
        """instantiate from YAML provided representation"""
        return cls(**seq)

    def to_content(self):
        return types.Content(parts=[types.Part(text=self.content)], role=self.role)


yaml.add_representer(Message, Message.represent_for_yaml)


CostConfig = collections.namedtuple("CostConfig", "name prompt_cost completion_cost")
CostCalculation = collections.namedtuple("CostCalculation", "name tokens cost")

costs = [
    CostConfig("gemini-2.0-flash-exp", 0.6, 0.3),
]


def get_tokens_and_costs(messages):
    return [
        CostCalculation(
            cost_config.name, *num_tokens_in_messages(messages, cost_config)
        )
        for cost_config in costs
    ]


def num_tokens_in_messages(messages, cost_config):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(f"{cost_config.name}")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    cost = 0
    for i, message in enumerate(messages):
        msg_tokens = (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        msg_tokens += len(encoding.encode(message.role))
        msg_tokens += len(encoding.encode(message.content))
        if i == len(messages) - 1 and message.role == MODEL_ROLE:
            cost += cost_config.completion_cost * msg_tokens
        else:
            cost += cost_config.prompt_cost * msg_tokens
        num_tokens += msg_tokens
    if messages[-1].role == "user":
        num_tokens += 2  # every reply is primed with <im_start>model
        cost += cost_config.prompt_cost * 2
    return num_tokens, cost / 1000000


def init_conversation(user_msg, system_msg: str | None = None):
    system = [Message("system", system_msg.strip())] if system_msg else []
    return system + [Message("user", user_msg)]


DEFAULT_GEMINI_SETTINGS = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "n": 1,
    "stream": True,
}


def map_from_stream(content_iter):
    """maps a gemini streaming generator a stream of Message with the
    final one being the completed Message"""
    role, message = None, ""
    gen_content: types.GenerateContentResponse
    for gen_content in content_iter:
        resp = gen_content.candidates[0].content
        if role is None:
            role = resp.role
        message += "\n".join(x.text for x in resp.parts)
        yield Message(role, message.strip())


def map_single(result: types.GenerateContentResponse):
    """maps a result to a Message"""
    response_message = result.candidates[0].content
    return Message(
        response_message.role, "\n".join(x.text for x in response_message.parts).strip()
    )


def build_client(config) -> genai.Client:
    assert config["api_key"] is not None, "api_key is required"
    return genai.Client(api_key=config["api_key"])


def build_genai_config(messages: list[Message], config) -> types.GenerateContentConfig:
    """Returns the system instruction from the config."""
    system_instruction = "\n".join(
        msg.content for msg in messages if msg.role == "system"
    ).strip()
    system_instruction = None if system_instruction == "" else system_instruction
    off = types.HarmBlockThreshold.OFF
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=config["temperature"],
        candidate_count=config["n"],
        safety_settings=[
            types.SafetySetting(category=category, threshold=off)
            for category in [
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            ]
        ],
    )


def query_chat(messages: list[Message], config) -> Message:
    """Queries the API with the given messages and config."""
    client = build_client(config)
    config = utils.merge_dicts(DEFAULT_GEMINI_SETTINGS, config)
    contents = [msg.to_content() for msg in messages if msg.role != "system"]
    try:
        kws = dict(
            model=config["model"],
            contents=contents,
            config=build_genai_config(messages, config),
        )
        if config["stream"]:
            result = client.models.generate_content_stream(**kws)
            return map_from_stream(result)
        else:
            result = client.models.generate_content(**kws)
            return map_single(result)
    except Exception as e:
        raise errors.ChatbladeError(f"error: {e}\n{traceback.format_exc()}")
