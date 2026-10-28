import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
MEMORY = ROOT / "memory.md"
SYSTEM_PROMPT = (
    """You are a helpful personal assistant.

    Here is what you should remember about the user from previous conversations:
    {memory}

    When you learn a new fact about the user, update the memory with save_memory.
    """
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file in the workspace directory. Creates the file if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file in the workspace directory. Fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Initial text to write. Omit this to create an empty file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_memory",
            "description": "Load the saved memory notes.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Replace the saved memory notes with the full text to remember.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete memory text to save.",
                    }
                },
                "required": ["content"],
            },
        },
    },
]


def list_files():
    return sorted(
        str(path.relative_to(WORKSPACE))
        for path in WORKSPACE.rglob("*")
        if path.is_file()
    )


def workspace_path(path):
    target = (WORKSPACE / path).resolve()
    if target == WORKSPACE.resolve() or not target.is_relative_to(WORKSPACE.resolve()):
        raise FileNotFoundError(path)
    return target


def read_file(path):
    target = workspace_path(path)
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_text()


def write_file(path, content):
    target = workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {target.relative_to(WORKSPACE)}"


def create_file(path, content=""):
    target = workspace_path(path)
    if target.exists():
        raise FileExistsError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"created {target.relative_to(WORKSPACE)}"


def load_memory():
    if not MEMORY.is_file():
        return ""
    return MEMORY.read_text()


def save_memory(content):
    MEMORY.write_text(content)
    return "saved memory"


def run_tool(name, args):
    if name == "list_files":
        return json.dumps(list_files())
    if name == "read_file":
        return read_file(args["path"])
    if name == "write_file":
        return write_file(args["path"], args["content"])
    if name == "create_file":
        return create_file(args["path"], args.get("content", ""))
    if name == "load_memory":
        return load_memory()
    if name == "save_memory":
        return save_memory(args["content"])
    raise ValueError(f"unknown tool: {name}")


def chat(user_message):
    api_key = "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"
    model = "openrouter/free"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())},
        {"role": "user", "content": user_message},
    ]
    while True:
        data = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
        }
        response = requests.post(base_url, headers=headers, json=data)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or ""
        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            print(f"tool: {name} {json.dumps(args)}", flush=True)
            try:
                result = run_tool(name, args)
                print(f"result: {result}", flush=True)
            except Exception as error:
                result = str(error)
                print(f"error: {result}", flush=True)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )


if __name__ == "__main__":
    print("Type a message and press Enter. Type 'quit' to exit.")
    while True:
        user_message = input("You: ").strip()
        if not user_message or user_message.lower() in {"quit", "exit"}:
            break
        print(chat(user_message))
        print()
