import asyncio
import json
from uuid import uuid4
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId
from simplechat import chatbot

from simplechat_agent import agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from db import db

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Models ------------------
class ChatRequest(BaseModel):
    message: str
    chat_id: str | None = None


# ------------------ Create Chat ------------------
@app.post("/chat/create")
async def create_chat():
    result = await db.chats.insert_one({"title": "New Chat", "messages": []})
    return {"chat_id": str(result.inserted_id)}


# ----------------------------------------------------------
#                AGENT STREAMING (FINAL WORKING)
# ----------------------------------------------------------
@app.get("/agent/stream")
async def agent_stream(message: str = Query(...), chat_id: str = Query(...)):
    # Validate chat_id
    try:
        oid = ObjectId(chat_id)
    except:
        return StreamingResponse(
            iter(["event: error\ndata: Invalid chat_id\n\n"]),
            media_type="text/event-stream"
        )

    chat = await db.chats.find_one({"_id": oid})
    if not chat:
        return StreamingResponse(
            iter(["event: error\ndata: Chat not found\n\n"]),
            media_type="text/event-stream"
        )

    # Build conversation history
    history = []
    for m in chat.get("messages", []):
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(SystemMessage(content=m["content"]))

    # Add user message
    history.append(HumanMessage(content=message))

    # Save user message in DB
    await db.chats.update_one(
        {"_id": oid},
        {"$push": {"messages": {"role": "user", "content": message}}}
    )

    # -----------------------------------------------------
    #                 SSE STREAMING LOOP
    # -----------------------------------------------------
    async def event_generator():
        try:
            async for step in agent.astream(input={"messages": history}):

                print("DEBUG step:", step)  # REMOVE after testing

                # --------------------------
                # TOOL CALL
                # --------------------------
                if "tool_call" in step:
                    yield (
                        "event: tool_call\n"
                        f"data: {json.dumps(step['tool_call'])}\n\n"
                    )

                # --------------------------
                # TOOL RESULT (OUTPUT)
                # --------------------------
                if "tool_result" in step:
                    yield (
                        "event: tool_output\n"
                        f"data: {json.dumps(step['tool_result'])}\n\n"
                    )

                # --------------------------
                # LLM NODE MESSAGE (AI Output)
                # --------------------------
                if "llm_node" in step:
                    msgs = step["llm_node"].get("messages", [])

                    for m in msgs:
                        content = getattr(m, "content", "")
                        if not content:
                            continue

                        # Send incremental assistant message to UI
                        yield (
                            "event: message\n"
                            f"data: {json.dumps({'content': content})}\n\n"
                        )

                        # Save in DB
                        await db.chats.update_one(
                            {"_id": oid},
                            {
                                "$push": {
                                    "messages": {
                                        "role": "agent",
                                        "content": content
                                    }
                                }
                            }
                        )

            # --------------------------
            # END STREAM
            # --------------------------
            yield "event: end\ndata: done\n\n"

        except Exception as e:
            print("STREAM ERROR:", e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------ Non-Streaming Chat ------------------
@app.post("/chat")
async def chat(payload: ChatRequest):
    if payload.chat_id is None:
        raise HTTPException(status_code=400, detail="chat_id is required")

    try:
        oid = ObjectId(payload.chat_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid chat_id")

    chat = await db.chats.find_one({"_id": oid})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    history = []
    for m in chat.get("messages", []):
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(SystemMessage(content=m["content"]))

    history.append(HumanMessage(content=payload.message))

    result = chatbot.invoke({"messages": history})
    reply = result["messages"][-1].content

    await db.chats.update_one(
        {"_id": oid},
        {"$push": {"messages": {"role": "user", "content": payload.message}}}
    )

    await db.chats.update_one(
        {"_id": oid},
        {"$push": {"messages": {"role": "agent", "content": reply}}}
    )

    return {"chat_id": payload.chat_id, "reply": reply}


# ------------------ Get All Chats ------------------
@app.get("/chats")
async def get_chats():
    return [
        {"id": str(chat["_id"]), "title": chat.get("title", "New Chat")}
        async for chat in db.chats.find()
    ]


# ------------------ Get Single Chat ------------------
@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    oid = ObjectId(chat_id)
    chat = await db.chats.find_one({"_id": oid})
    return {
        "id": str(chat["_id"]),
        "title": chat.get("title", "New Chat"),
        "messages": chat.get("messages", []),
    }


# ------------------ Delete Chat ------------------
@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    oid = ObjectId(chat_id)
    await db.chats.delete_one({"_id": oid})
    return {"success": True}
