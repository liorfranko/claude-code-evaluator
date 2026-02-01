#!/usr/bin/env python3
"""Simple test to validate developer agent Q&A functionality."""

import asyncio
import logging
import os
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from claude_evaluator.agents.developer import DeveloperAgent, SDK_AVAILABLE
from claude_evaluator.models.question import QuestionContext, QuestionItem, QuestionOption


async def test_developer_answer_question():
    """Test the developer agent's answer_question method."""

    print("=" * 60)
    print("Testing Developer Agent Q&A")
    print("=" * 60)

    # Check SDK availability
    print(f"\nSDK Available: {SDK_AVAILABLE}")
    if not SDK_AVAILABLE:
        print("ERROR: claude-agent-sdk is not installed!")
        return False

    # Create developer agent - try different models
    # claude-3-haiku@20240307 returns 404 - model not found
    # Try claude-haiku-4-5@20251001 instead
    developer = DeveloperAgent(
        cwd=os.getcwd(),  # Use current directory
        developer_qa_model="claude-haiku-4-5@20251001",
    )

    print(f"\nDeveloper Agent created:")
    print(f"  - cwd: {developer.cwd}")
    print(f"  - model: {developer.developer_qa_model}")

    # Create a simple question context simulating worker asking a question
    question_context = QuestionContext(
        questions=[
            QuestionItem(
                question="Should I proceed with the implementation?",
                header="Confirmation",
                options=[
                    QuestionOption(label="Yes", description="Proceed with implementation"),
                    QuestionOption(label="No", description="Cancel and wait"),
                ],
            )
        ],
        conversation_history=[
            {"role": "user", "content": "Create a hello world function"},
            {"role": "assistant", "content": "I'll create a hello world function for you."},
        ],
        session_id="test-session",
        attempt_number=1,
    )

    print(f"\nQuestion Context:")
    print(f"  - Questions: {len(question_context.questions)}")
    print(f"  - Question: {question_context.questions[0].question}")
    print(f"  - History messages: {len(question_context.conversation_history)}")

    # Try to answer the question
    print("\n" + "-" * 60)
    print("Calling developer.answer_question()...")
    print("-" * 60)

    try:
        result = await developer.answer_question(question_context)
        print("\n✅ SUCCESS!")
        print(f"  - Answer: {result.answer}")
        print(f"  - Model used: {result.model_used}")
        print(f"  - Context size: {result.context_size}")
        print(f"  - Generation time: {result.generation_time_ms}ms")
        return True

    except Exception as e:
        print(f"\n❌ FAILED!")
        print(f"  - Exception type: {type(e).__name__}")
        print(f"  - Exception message: {str(e)}")

        # Print full traceback
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        return False


async def test_sdk_query_directly():
    """Test the SDK query function directly to isolate the issue."""

    print("\n" + "=" * 60)
    print("Testing SDK query directly")
    print("=" * 60)

    try:
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
    except ImportError as e:
        print(f"ERROR: Could not import claude_agent_sdk: {e}")
        return False

    cwd = os.getcwd()
    model = "claude-haiku-4-5@20251001"  # claude-3-haiku@20240307 returns 404
    prompt = "Respond with just the word 'hello'"

    print(f"\nTest parameters:")
    print(f"  - cwd: {cwd}")
    print(f"  - model: {model}")
    print(f"  - prompt: {prompt}")

    print("\n" + "-" * 60)
    print("Calling sdk_query()...")
    print("-" * 60)

    try:
        result_message = None
        query_gen = sdk_query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=cwd,
                model=model,
                max_turns=1,
                permission_mode="plan",
            ),
        )

        try:
            async for message in query_gen:
                msg_type = type(message).__name__
                print(f"  Received message type: {msg_type}")
                if msg_type == "ResultMessage":
                    result_message = message
        finally:
            await query_gen.aclose()

        if result_message:
            print("\n✅ SUCCESS!")
            print(f"  - Result: {result_message.result if hasattr(result_message, 'result') else 'N/A'}")
            return True
        else:
            print("\n⚠️ No ResultMessage received")
            return False

    except Exception as e:
        print(f"\n❌ FAILED!")
        print(f"  - Exception type: {type(e).__name__}")
        print(f"  - Exception message: {str(e)}")

        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEVELOPER Q&A TEST")
    print("=" * 60)

    # Run both tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # First test SDK directly
        sdk_ok = loop.run_until_complete(test_sdk_query_directly())

        # Then test developer agent
        dev_ok = loop.run_until_complete(test_developer_answer_question())

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  SDK query direct: {'✅ PASS' if sdk_ok else '❌ FAIL'}")
        print(f"  Developer Q&A:    {'✅ PASS' if dev_ok else '❌ FAIL'}")

        sys.exit(0 if (sdk_ok and dev_ok) else 1)

    finally:
        loop.close()
