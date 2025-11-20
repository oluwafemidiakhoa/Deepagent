"""
Test OpenAI API Key

Quick test to verify your OpenAI API key is working.
"""

import os

def test_openai_key():
    print("\n" + "="*80)
    print("TESTING OPENAI API KEY")
    print("="*80 + "\n")

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return False

    print(f"API Key (masked): {api_key[:20]}...{api_key[-10:]}")
    print(f"Key length: {len(api_key)} characters\n")

    try:
        import openai
        print("[OK] OpenAI library imported")
    except ImportError:
        print("[ERROR] OpenAI library not installed")
        print("Install with: pip install openai")
        return False

    try:
        # Test with simple completion
        print("\nTesting API connection with simple completion...")

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API key is working!' in exactly 5 words."}
            ],
            max_tokens=20,
            temperature=0
        )

        result = response.choices[0].message.content
        print(f"\n[SUCCESS] API Response: {result}")
        print(f"Model used: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens}")

        return True

    except openai.AuthenticationError as e:
        print(f"\n[ERROR] Authentication failed!")
        print(f"Error: {e}")
        print("\nPossible issues:")
        print("  - API key is invalid or expired")
        print("  - API key doesn't have proper permissions")
        print("  - Check https://platform.openai.com/api-keys")
        return False

    except openai.RateLimitError as e:
        print(f"\n[ERROR] Rate limit exceeded!")
        print(f"Error: {e}")
        print("\nYou've hit the API rate limit. Wait and try again.")
        return False

    except openai.APIError as e:
        print(f"\n[ERROR] OpenAI API error!")
        print(f"Error: {e}")
        return False

    except Exception as e:
        print(f"\n[ERROR] Unexpected error!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = test_openai_key()

    if success:
        print("\n" + "="*80)
        print("API KEY IS WORKING!")
        print("="*80)
        print("\nYou can now use DeepAgent with real OpenAI models:")
        print("\n  from deepagent.core.self_editing_agent import create_self_editing_agent")
        print("\n  agent = create_self_editing_agent(")
        print("      llm_provider='openai',")
        print("      llm_model='gpt-4',")
        print("      enable_learning=True")
        print("  )")
        print("\n  result = agent.execute_with_learning('Your task here')")
        print("\nOr run the SEAL examples:")
        print("  python examples/seal_learning_example.py")

    else:
        print("\n" + "="*80)
        print("API KEY TEST FAILED")
        print("="*80)
        print("\nPlease check your API key and try again.")
        print("Get your key at: https://platform.openai.com/api-keys")

    print()
