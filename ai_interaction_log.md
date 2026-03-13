### Task: Task 1A: Page Setup & API Connection
**Prompt:** 
I am building a chatbot using streamlit and the Hugging Face API. 

Let's first setup the layout of our chat app. Here are the requirements.
Use st.set_page_config(page_title="My AI Chat", layout="wide").
Load your Hugging Face token using st.secrets["HF_TOKEN"]. The token must never be hardcoded in app.py.
If the token is missing or empty, display a clear error message in the app. The app must not crash.
Send a single hardcoded test message (e.g. "Hello!") to the Hugging Face API using the loaded token and display the model’s response in the main area.
Handle API errors gracefully (missing token, invalid token, rate limit, network failure) with a user-visible message rather than a crash.
**AI Suggestion:** AI implemented the instructions provided in the prompt. AI chose "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` and reload the app" as the error message when a token is missing.
**My Modifications & Reflections:** I verified that the code worked by running the app first with a secrets.toml file, and then moving the secrets.toml outside of .streamlit/ and running again. The app showed an error message instead of crashing, which was what I wanted.

### Task: Task 1B: Multi-Turn Conversation UI
**Prompt:**
Great! Now let's extend our functionality to take real user input. Here are the requirements:
Extend Part A to replace the hardcoded test message with a real input interface.
Use native Streamlit chat UI elements. Render messages with st.chat_message(...) and collect user input with st.chat_input(...).
Add a fixed input bar at the bottom of the main area.
Store the full conversation history in st.session_state. After each exchange, append both the user message and the assistant response to the history.
Send the full message history with each API request so the model maintains context.
Render the conversation history above the input bar using default Streamlit UI elements rather than CSS-based custom chat bubbles.
The message history must scroll independently of the input bar — the input bar stays visible at all times.

Notice that we are replacing the hardcoded "Hello" message we used previously, while keeping everything else the same.
**AI Suggestion:** AI implemented the instructions provided in the prompt and replaced the hardcoded "Hello" message to a chat interface with a main area and input bar. 
**My Modifications & Reflections:** I verified that the code was correct by first asking the chatbot, "Hello! Can you tell me how to cook pasta?" and then "what should go with it?" I deliberately chose the second question to be vague to test if the chatbot remembered what we were talking about. The chatbot output a discussion on various pasta sauces in response to the second question, showing that the chatbot has the context of previous messages. Additionally, the main chat area allowed scrolling, but the input bar is fixed at the bottom.

### Task: Task 1C: Chat Management
**Prompt:**
Awesome! Now we are going to further update our program to allow the user to add a new chat. Here are the requirements:
Add a New Chat button to the sidebar that creates a fresh, empty conversation and adds it to the sidebar chat list.
Use the native Streamlit sidebar (st.sidebar) for chat navigation.
The sidebar shows a scrollable list of all current chats, each displaying a title and timestamp.
The currently active chat must be visually highlighted in the sidebar.
Clicking a chat in the sidebar switches to it without deleting or overwriting any other chats.
Each chat entry must have a ✕ delete button. Clicking it removes the chat from the list. If the deleted chat was active, the app must switch to another chat or show an empty state.

Remember to use native Streamlit elements and not CSS. Furthermore, there is a "Status" column from earlier, but I don't need that. Feel free to replace this "Status" column in the app with the sidebar, or use the space as you think appropriate.
**AI Suggestion:** AI followed the instructions given in the prompt above. Each chat in the sidebar will be named with the user's first message in the chat, and if a current chat is deleted the app will go to the nearest remaining chat.
**My Modifications & Reflections:**
I entered this prompt to Codex to make changes:
"Let's implement these minor changes:
1) If there are no chats yet, allow the user to create a chat by simply typing a prompt into the input box. In other words, if it is the first chat, the user should be able to start chatting immediately without having to press the "New Chat" button. 
2) When the user submits an input, the user's input should immediately show up on the main chat area, and a "Thinking" message should appear while Hugging Face API is responding."
I verified that the code worked by creating multiple chats, deleting chats, and making sure active chats are highlighted.

### Task: Task 1D: Chat Persistence
**Prompt:**
Awesome! Let's update our program once more to save the chats the user has with the chatbot. I've created a chats/ directory for this. Here are the requirements:
Each chat session is saved as a separate JSON file inside a chats/ directory. Each file must store at minimum: a chat ID, a title or timestamp, and the full message history.
On app startup, all existing files in chats/ are loaded and shown in the sidebar automatically.
Returning to a previous chat and continuing the conversation must work correctly.
Deleting a chat (✕ button) must also delete the corresponding JSON file from chats/.
A generated or summarized chat title is acceptable and encouraged. The title does not need to be identical to the first user message.
**AI Suggestion:** AI implemented the instructions provided in the prompt above. Saved chat titles will be local summaries of the messages.
**My Modifications & Reflections:** I verified that the code was correct by creating new chats, deleting chats, and checking to see if json files under the chats/ folder are being created/deleted in real time. Additionally, I closed the app and reran it in the terminal and confirmed that previous chats were loaded and continuable (meaning, when I send a vague follow up question the chatbot can reply with context of previous messages).

### Task: Task 2: Response Streaming
**Prompt:**
Awesome! Now let's add some more functionality. I want to update the chatbot app so that the bot's reply is displayed token-by-token as it is generated. Here are the requirements:
Use the stream=True parameter in your API request and handle the server-sent event stream.
In Streamlit, use native Streamlit methods such as st.write_stream() or manually update a placeholder with st.empty() as chunks arrive.
The full streamed response must be saved to the chat history once streaming is complete.
Hint: Add stream=True to your request payload and set stream=True on the requests.post() call. The response body will be a series of data: lines in SSE format.

Note that we are just updating the chatbot response to be streamed while keeping everything else the same.
**AI Suggestion:** AI followed the instructions provided in the prompt above, displaying the chatbot output token-by-token, and saving the message once the entire output has been generated.
**My Modifications & Reflections:** 
I entered this prompt to make changes:
"In the UI, the streaming is too fast to be noticed due to how fast the chatbot API model is in Hugging Face. I want to add a very short delay between rendering chunks so that the streaming behavior is visible in the app."
I then verified that the code was correct by checking to see if chatbot responses appear incrementally and if the chat history is still saved in a json under chats/ folder.

### Task: Task 3: User Memory
**Prompt:** 
Great! Finally, let's also implement more advanced user memory and store user preferences from conversations. Here are the requirements:
After each assistant response, make a second lightweight API call asking the model to extract any personal traits or preferences mentioned by the user in that message.
Extracted traits are stored in a memory.json file. Example categories might include name, preferred language, interests, communication style, favorite topics, or other useful personal preferences.
The sidebar displays a User Memory expander panel showing the currently stored traits.
Include a native Streamlit control to clear/reset the saved memory.
Stored memory is injected into the system prompt of future conversations so the model can personalize responses.
Hint: A simple memory extraction prompt might look like: “Given this user message, extract any personal facts or preferences as a JSON object. If none, return {}”

Note that we will always be using native streamlit components and not CSS. Please ask me any clarifications questions you need.
**AI Suggestions:** AI followed the instructions provided in the prompt above and also asked some clarifying questions. The answers to those questions are: 1. Saved user memory should apply across all chats (global memory); 2. If newly extracted memory conflicts with what's already in memory.json, the app should use the newest memory.
**My Modifications & Reflections:** 
I entered the following promps to make changes:
"The user memory sidebar shouldn't display user preferences in a json format. The json format should be used when actually storing into memory.json, but let's use a more human-friendly, readable format for the user memory sidebar. Additionally, even when there's no existing user memory, the sidebar should still be there, but it can be empty."
"Good. Also, the "name" category means what the user's name is, if the user ever chooses to provide their name to the chatbot. It should be empty if the user hasn't provided their name."
I also asked Codex to make the json memory extraction part of the program more robust so that I won't get the error, "Memory extraction returned non-JSON content."
Here is another debugging prompt I wrote to Codex:
"The user memory extraction is somehow not that great anymore. Take a look at this prompt given by the user: "Hi, my name is Minlan. I enjoy the outdoors and love running in the garden at my university. Can you create a running plan for me to improve my speed and distance?"
Somehow only the name of the user was extracted, ignoring other interests such as running and outdoors."
After all this debugging, the user memory feature was still very buggy. Either it would return false information, or it wouldn't return enough information. So, I entered this prompt to codex:
"Hmm, let's actually change our memory extraction feature to be a little bit more specific. Here are the categories of traits to look for: Name (user's name), Interests, Communication Style, and Hobbies/Activities. Limit the search to only these traits."
Finally, I verified that the code was correct by putting example user prompts and checking that extracted user memory is accurate. I checked that the user memory applied to all future chats by creating a new chat and asking the chatbot what my name was. I also made sure that the clear memory function worked by asking the chatbot what my name was after clearing the memory. Finally, I verified that memory.json was correctly updated as the user enters new interests/hobbies or clears previous memory. Closing the app and reopening also displays saved user memory along with saved user chats.

I also asked Codex to fix the chatbot app layout with the following prompt:
"Let's do a quick fix of the app layout. currently, the webpage automatically scrolls all the way down to the bottom of the chat area, meaning the user can't see the chat title. Can you make it so that the user can see the chat title AND the input text box? Can you also make it so that the user can see both the chat history sidebar AND the user memory sidebar? Finally, let's order the chat history chats from most recent to least recent, and the app should go to the most recent chat when first loaded."
"Let's let the user be able to see the streamlit "ECS32A Chatbot" title, as well."

Prompt to make some final changes to the app:
"Great! Let's make some final changes to the app. First, let's put the "Clear Memory" button in the user memory at the top of the user memory section instead of at the bottom. Second, let's remove the "Streamlit + Hugging Face chat interface" caption and the chat title, so that the user just sees the "ECS32A Chatbot" title above the main chat area. Finally, let's change the chat title function a little bit to show chat titles in the sidebar that is a summary of the user's first message instead of just the first few words of the first message."