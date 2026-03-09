"""Telegram bot that answers questions about chat history using RAG."""

import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import TELEGRAM_BOT_TOKEN
from rag import RAG

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

rag: RAG | None = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm a RAG chatbot for your Telegram chat history.\n"
        "Ask me anything about the conversations and I'll search through the messages to answer.\n\n"
        "Just type your question!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    if not query:
        return

    logger.info(f"Query from {update.effective_user.username}: {query}")

    await update.message.chat.send_action("typing")

    try:
        answer = rag.answer(query)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        await update.message.reply_text(
            "Sorry, something went wrong while processing your question. Please try again."
        )


def main():
    global rag

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set. Check your .env file.")

    logger.info("Initializing RAG system...")
    rag = RAG()
    logger.info("RAG system ready.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()

    rag.close()


if __name__ == "__main__":
    main()
