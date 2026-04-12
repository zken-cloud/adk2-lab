import asyncio
from typing import AsyncGenerator, Union

from google.adk.agents.llm_agent import LlmAgent

from src.prompts import GENERATE_BLOG_POST_PROMPT
from src.tools import post_to_platform


generate_blog_post_agent = LlmAgent(
    name="generate_blog_post_agent",
    model="gemini-2.5-flash",
    instruction=GENERATE_BLOG_POST_PROMPT,
)