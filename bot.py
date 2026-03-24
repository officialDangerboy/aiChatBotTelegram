import os
import asyncio
import logging
import threading
from collections import defaultdict
from dotenv import load_dotenv

from flask import Flask
from groq import Groq
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PORT = int(os.getenv("PORT", 10000))
MODEL = "llama-3.3-70b-versatile"
MEMORY_LIMIT = 20

# Initialize Flask for health checks
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running", 200

@app.route('/health')
def health():
    return {"status": "OK"}, 200

def run_flask():
    app.run(host='0.0.0.0', port=PORT)

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# In-memory storage for user history
user_memory = defaultdict(list)

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def get_ai_response(user_id, user_message):
    """Handles Groq API calls with error wrapping."""
    system_prompt = (
        "You are a helpful AI assistant. Always use **bold text** for emphasis. "
        "Use bullet points (•) for lists. Provide code snippets in code blocks. "
        "Do NOT use '##' headers."
    )
    
    # Update memory
    user_memory[user_id].append({"role": "user", "content": user_message})
    
    # Keep only the last N messages
    if len(user_memory[user_id]) > MEMORY_LIMIT:
        user_memory[user_id] = user_memory[user_id][-MEMORY_LIMIT:]

    messages = [{"role": "system", "content": system_prompt}] + user_memory[user_id]

    try:
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        response = completion.choices[0].message.content
        user_memory[user_id].append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        logging.error(f"Groq API Error: {e}")
        return "I'm sorry, I encountered an error processing your request."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears history and greets the user."""
    user_id = update.effective_user.id
    first_name = update.effective_user.first_name
    user_memory[user_id] = [] # Clear history
    
    await update.message.reply_text(f"Hello **{first_name}**! Your history has been reset. How can I help you today?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes incoming messages with typing indicator."""
    user_id = update.effective_user.id
    text = update.message.text

    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)

    # Get AI response
    response_text = await get_ai_response(user_id, text)

    try:
        # Attempt Markdown V1 (standard) or fallback to plain text
        await update.message.reply_text(response_text, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        logging.warning(f"Markdown parsing failed: {e}")
        await update.message.reply_text(response_text)

async def main():
    """Main entry point using asyncio.run() compatibility."""
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Build Telegram Application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Add Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logging.info("Starting bot...")
    
    # Start polling
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        # Keep the event loop running
        await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass