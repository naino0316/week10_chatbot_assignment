from datetime import datetime
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import requests
import streamlit as st

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHAT_HISTORY_HEIGHT = 360
SIDEBAR_LIST_HEIGHT = 220
CHATS_DIR = Path("chats")
MEMORY_FILE = Path("memory.json")
STREAM_RENDER_DELAY_SECONDS = 0.02
MEMORY_SYSTEM_PROMPT_PREFIX = (
    "Use this saved user memory to personalize responses when relevant: "
)
ALLOWED_MEMORY_KEYS = {
    "name",
    "interests",
    "communication_style",
    "hobbies_activities",
}
MEMORY_EXTRACTION_PROMPT = (
    "You extract user memory from a single user message. Return a JSON object only. "
    "Extract only these categories when they are explicitly stated by the user: "
    "name, interests, communication_style, hobbies_activities. Do not return any "
    "other keys. The 'name' field means the user's actual name only if the user "
    "explicitly states it; otherwise omit 'name'. Use arrays for interests and "
    "hobbies_activities when there are multiple items. If nothing useful is present, "
    "return {}. Do not guess missing facts. Example "
    "output: {\"name\":,\"interests\":[],"
    "\"hobbies_activities\":[]}"
)


class ChatStreamError(Exception):
    pass


def normalize_json_text(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`").strip()
        if normalized.lower().startswith("json"):
            normalized = normalized[4:].strip()
    return normalized


def extract_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def extract_memory_fallback(user_message: str) -> dict:
    extracted = {}
    message = " ".join(user_message.split())

    name_match = re.search(
        r"\b(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z'-]+)\b",
        user_message,
        re.IGNORECASE,
    )
    if name_match:
        extracted["name"] = name_match.group(1).strip().title()

    interests = []
    hobbies_activities = []
    seen_interests = set()
    seen_hobbies = set()

    def normalize_memory_item(value: str) -> str:
        cleaned = value.strip(" .,!?:;").lower()
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        cleaned = re.sub(r"^(love|enjoy|like)\s+", "", cleaned)
        return cleaned

    def add_interest(value: str) -> None:
        cleaned = normalize_memory_item(value)
        if not cleaned or cleaned in seen_interests:
            return
        seen_interests.add(cleaned)
        interests.append(cleaned)

    def add_hobby(value: str) -> None:
        cleaned = normalize_memory_item(value)
        if not cleaned or cleaned in seen_hobbies:
            return
        seen_hobbies.add(cleaned)
        hobbies_activities.append(cleaned)

    interest_patterns = [
        r"\bi enjoy ([^.?!]+)",
        r"\bi like ([^.?!]+)",
        r"\bi love ([^.?!]+)",
        r"\bi'm into ([^.?!]+)",
        r"\bmy hobbies include ([^.?!]+)",
    ]

    for pattern in interest_patterns:
        for match in re.finditer(pattern, user_message, re.IGNORECASE):
            phrase = match.group(1)
            parts = re.split(r",| and ", phrase)
            for part in parts:
                cleaned = part.strip(" .,!?:;")
                if cleaned.lower().startswith("to "):
                    continue
                cleaned = re.sub(r"\bin the .*$", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\bat the .*$", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\bto improve .*$", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\bfor me .*$", "", cleaned, flags=re.IGNORECASE)
                if cleaned:
                    normalized = normalize_memory_item(cleaned)
                    if re.search(
                        r"\brunn?ing\b|\bhik(?:e|ing)\b|\bwalk(?:ing)?\b|\bcycl(?:ing|e)\b|\bswimm(?:ing)?\b|\btennis\b|\bgardening\b",
                        normalized,
                    ):
                        add_hobby(normalized)
                    else:
                        add_interest(normalized)

    if re.search(r"\boutdoors?\b", message, re.IGNORECASE):
        add_interest("outdoors")
    if re.search(r"\brunn(?:ing|er)?\b|\brun\b", message, re.IGNORECASE):
        add_hobby("running")
    if re.search(r"\bhik(?:e|ing)\b", message, re.IGNORECASE):
        add_hobby("hiking")
    if re.search(r"\bgarden(?:ing)?\b", message, re.IGNORECASE):
        add_hobby("gardening")

    communication_style_patterns = [
        r"\bplease be ([^.?!]+)",
        r"\bi prefer ([^.?!]+) responses",
        r"\bkeep it ([^.?!]+)",
        r"\bexplain (?:things )?in a ([^.?!]+) way",
        r"\btalk to me like ([^.?!]+)",
    ]
    for pattern in communication_style_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            style = match.group(1).strip(" .,!?:;").lower()
            if style:
                extracted["communication_style"] = style
                break

    if interests:
        extracted["interests"] = interests
    if hobbies_activities:
        extracted["hobbies_activities"] = hobbies_activities

    return extracted


def filter_memory_fields(memory: dict) -> dict:
    filtered = {}
    for key, value in memory.items():
        if key not in ALLOWED_MEMORY_KEYS:
            continue
        if value in ("", None, [], {}):
            continue
        filtered[key] = value
    return filtered


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def extract_stream_text(event: dict) -> str:
    choices = event.get("choices", [])
    if not choices:
        return ""

    delta = choices[0].get("delta", {})
    content = delta.get("content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


def build_chat_messages(memory: dict, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not memory:
        return list(messages)

    system_message = {
        "role": "system",
        "content": MEMORY_SYSTEM_PROMPT_PREFIX + json.dumps(memory, ensure_ascii=True),
    }
    return [system_message, *messages]


def stream_chat_response(token: str, messages: list[dict[str, str]]):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
            stream=True,
        )
    except requests.exceptions.Timeout:
        raise ChatStreamError("The request to Hugging Face timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ChatStreamError(
            "Could not connect to Hugging Face. Check your network connection and try again."
        )
    except requests.exceptions.RequestException as exc:
        raise ChatStreamError(f"Unexpected request error: {exc}") from exc

    if response.status_code == 401:
        response.close()
        raise ChatStreamError("Your Hugging Face token appears to be invalid.")
    if response.status_code == 429:
        response.close()
        raise ChatStreamError(
            "The Hugging Face API rate limit was reached. Please wait a bit and try again."
        )
    if response.status_code >= 400:
        try:
            details = response.json()
        except ValueError:
            details = response.text or "No additional details were returned."
        response.close()
        raise ChatStreamError(
            f"Hugging Face API error ({response.status_code}): {details}"
        )

    def event_stream():
        received_text = False

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue

                data_line = raw_line[5:].strip()
                if data_line == "[DONE]":
                    break

                try:
                    event = json.loads(data_line)
                except json.JSONDecodeError as exc:
                    raise ChatStreamError(
                        "The Hugging Face API returned malformed streaming data."
                    ) from exc

                text_chunk = extract_stream_text(event)
                if not text_chunk:
                    continue

                received_text = True
                time.sleep(STREAM_RENDER_DELAY_SECONDS)
                yield text_chunk
        finally:
            response.close()

        if not received_text:
            raise ChatStreamError(
                "The Hugging Face API returned success, but no assistant message was found."
            )

    return event_stream()


def fetch_memory_update(token: str, user_message: str) -> tuple[dict | None, str | None]:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        return None, "Memory extraction timed out."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect while updating user memory."
    except requests.exceptions.RequestException as exc:
        return None, f"Memory extraction request error: {exc}"

    if response.status_code >= 400:
        return None, f"Memory extraction failed ({response.status_code})."

    try:
        data = response.json()
    except ValueError:
        return None, "Memory extraction returned invalid JSON."

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not isinstance(content, str) or not content.strip():
        return {}, None

    normalized_content = normalize_json_text(content)

    try:
        extracted = json.loads(normalized_content)
    except json.JSONDecodeError:
        object_text = extract_json_object_text(normalized_content)
        if not object_text:
            fallback_memory = extract_memory_fallback(user_message)
            if fallback_memory:
                return fallback_memory, "Memory updated using fallback extraction."
            return None, "Memory extraction returned non-JSON content."
        try:
            extracted = json.loads(object_text)
        except json.JSONDecodeError:
            fallback_memory = extract_memory_fallback(user_message)
            if fallback_memory:
                return fallback_memory, "Memory updated using fallback extraction."
            return None, "Memory extraction returned invalid JSON content."

    if not isinstance(extracted, dict):
        fallback_memory = extract_memory_fallback(user_message)
        if fallback_memory:
            return fallback_memory, "Memory updated using fallback extraction."
        return None, "Memory extraction did not return a JSON object."

    return filter_memory_fields(extracted), None


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp).strftime("%b %d, %I:%M %p")


def summarize_chat_title(text: str, limit: int = 40) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return "New Chat"

    sentences = re.split(r"[.?!]+", normalized)
    candidates = [sentence.strip() for sentence in sentences if sentence.strip()]
    summary = candidates[0] if candidates else normalized

    if len(candidates) > 1:
        for candidate in candidates:
            if re.search(
                r"\b(can you|could you|please|help me|i need|i want|create|make|plan)\b",
                candidate,
                re.IGNORECASE,
            ):
                summary = candidate
                break

    summary = re.sub(
        r"^(hi|hello|hey)\b[,!\s]*",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    summary = re.sub(
        r"^(can you|could you|would you|please|help me|i need|i want to)\b\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    ).strip(" ,")

    if not summary:
        summary = normalized

    summary = summary[:1].upper() + summary[1:]
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3].rstrip()}..."


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}
    return filter_memory_fields(data)


def save_memory(memory: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_memory() -> None:
    save_memory({})


def merge_memory_value(existing, new_value):
    if isinstance(existing, list) or isinstance(new_value, list):
        merged = []
        for item in existing if isinstance(existing, list) else [existing]:
            if item not in merged:
                merged.append(item)
        for item in new_value if isinstance(new_value, list) else [new_value]:
            if item not in merged:
                merged.append(item)
        return merged
    return new_value


def merge_memory(existing_memory: dict, new_memory: dict) -> dict:
    merged = dict(filter_memory_fields(existing_memory))

    for key, value in filter_memory_fields(new_memory).items():
        if value in ("", None, [], {}):
            continue
        if key in merged:
            merged[key] = merge_memory_value(merged[key], value)
        else:
            merged[key] = value

    return merged


def format_memory_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def render_memory_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def create_chat() -> dict:
    timestamp = current_timestamp()
    return {
        "id": str(uuid4()),
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def sort_chats_by_recent() -> None:
    st.session_state["chats"].sort(
        key=lambda chat: chat.get("updated_at", chat.get("created_at", "")),
        reverse=True,
    )


def chat_file_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat: dict) -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    chat_file_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_chats() -> list[dict]:
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []

    for path in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(chat, dict):
            continue

        required_fields = {"id", "title", "created_at", "updated_at", "messages"}
        if not required_fields.issubset(chat):
            continue
        if not isinstance(chat["messages"], list):
            continue

        chats.append(chat)

    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    return chats


def delete_chat_file(chat_id: str) -> None:
    try:
        chat_file_path(chat_id).unlink(missing_ok=True)
    except OSError:
        pass


def get_active_chat() -> dict | None:
    active_chat_id = st.session_state["active_chat_id"]
    for chat in st.session_state["chats"]:
        if chat["id"] == active_chat_id:
            return chat
    return None


def add_new_chat() -> None:
    chat = create_chat()
    st.session_state["chats"].append(chat)
    sort_chats_by_recent()
    st.session_state["active_chat_id"] = chat["id"]
    save_chat(chat)


def delete_chat(chat_id: str) -> None:
    chats = st.session_state["chats"]
    delete_index = next(
        (index for index, chat in enumerate(chats) if chat["id"] == chat_id),
        None,
    )
    if delete_index is None:
        return

    was_active = st.session_state["active_chat_id"] == chat_id
    chats.pop(delete_index)
    delete_chat_file(chat_id)

    if not chats:
        st.session_state["active_chat_id"] = None
        return

    if was_active:
        next_index = min(delete_index, len(chats) - 1)
        st.session_state["active_chat_id"] = chats[next_index]["id"]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", use_container_width=True):
            add_new_chat()
            st.rerun()

        chat_list = st.container(height=SIDEBAR_LIST_HEIGHT)
        with chat_list:
            if not st.session_state["chats"]:
                st.info("No chats yet. Create one to get started.")

            for chat in st.session_state["chats"]:
                row_left, row_right = st.columns([5, 1])
                is_active = chat["id"] == st.session_state["active_chat_id"]

                with row_left:
                    if st.button(
                        chat["title"],
                        key=f"select_{chat['id']}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["active_chat_id"] = chat["id"]
                        st.rerun()
                    st.caption(format_timestamp(chat["updated_at"]))

                with row_right:
                    if st.button(
                        "\u2715",
                        key=f"delete_{chat['id']}",
                        use_container_width=True,
                    ):
                        delete_chat(chat["id"])
                        st.rerun()

        with st.expander("User Memory", expanded=True):
            if st.button("Clear Memory", use_container_width=True):
                st.session_state["memory"] = {}
                st.session_state["memory_notice"] = None
                clear_memory()
                st.rerun()

            if st.session_state["memory"]:
                for key, value in st.session_state["memory"].items():
                    st.write(
                        f"**{format_memory_label(key)}:** {render_memory_value(value)}"
                    )

            if st.session_state["memory_notice"]:
                st.caption(st.session_state["memory_notice"])


st.set_page_config(page_title="ECS32A Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ECS32A Chatbot")

token = load_hf_token()

if "chats" not in st.session_state:
    st.session_state["chats"] = load_chats()
if "active_chat_id" not in st.session_state:
    st.session_state["active_chat_id"] = (
        st.session_state["chats"][0]["id"] if st.session_state["chats"] else None
    )
if "memory" not in st.session_state:
    st.session_state["memory"] = load_memory()
if "memory_notice" not in st.session_state:
    st.session_state["memory_notice"] = None

active_chat_ids = {chat["id"] for chat in st.session_state["chats"]}
if st.session_state["active_chat_id"] not in active_chat_ids:
    st.session_state["active_chat_id"] = (
        st.session_state["chats"][0]["id"] if st.session_state["chats"] else None
    )

render_sidebar()

active_chat = get_active_chat()
prompt_disabled = not token

if active_chat is None:
    st.info("Start typing below to begin your first chat, or create one from the sidebar.")

if not token:
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` and reload the app."
    )

chat_history = st.container(height=CHAT_HISTORY_HEIGHT)
with chat_history:
    if active_chat is not None:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

prompt = st.chat_input(
    "Type a message and press ENTER",
    disabled=prompt_disabled,
)

if prompt:
    if active_chat is None:
        add_new_chat()
        active_chat = get_active_chat()

    user_message = {"role": "user", "content": prompt}
    active_chat["messages"].append(user_message)
    active_chat["updated_at"] = current_timestamp()
    sort_chats_by_recent()

    if active_chat["title"] == "New Chat":
        active_chat["title"] = summarize_chat_title(prompt)

    save_chat(active_chat)

    with chat_history:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                assistant_reply = st.write_stream(
                    stream_chat_response(
                        token,
                        build_chat_messages(
                            st.session_state["memory"], active_chat["messages"]
                        ),
                    )
                )
            except ChatStreamError as error:
                assistant_reply = None
                st.error(str(error))

    if assistant_reply:
        assistant_message = {"role": "assistant", "content": assistant_reply}
        active_chat["messages"].append(assistant_message)
        active_chat["updated_at"] = current_timestamp()
        sort_chats_by_recent()
        save_chat(active_chat)

        extracted_memory, memory_error = fetch_memory_update(token, prompt)
        if extracted_memory:
            st.session_state["memory"] = merge_memory(
                st.session_state["memory"], extracted_memory
            )
            save_memory(st.session_state["memory"])
            st.session_state["memory_notice"] = "User memory updated."
        elif memory_error:
            st.session_state["memory_notice"] = memory_error
        else:
            st.session_state["memory_notice"] = None

    st.rerun()
