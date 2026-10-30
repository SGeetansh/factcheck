import argparse
import os
import sys
from typing import cast

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
# Default to the standard LiteLLM Proxy URL if not set in env
LITELLM_BASE_URL = os.getenv("LLM_API_BASE")
# Alternatively, use "http://localhost:4000" if running locally
API_KEY = os.getenv("LLM_API_KEY")


def print_guide() -> None:
    """Print the usage guide requested by the `--guide` flag."""
    guide_text = """
    ===========================================
    💰 LiteLLM Budget Checker - Usage Guide
    ===========================================
    
    This tool helps you monitor your API usage against your allocated budget.
    
    Setup:
    1. Ensure your .env file contains:
       LLM_API_KEY=sk-...
       LLM_API_BASE=https://recllm.brahmastra.tech/
       
    How Cost Tracking Works:
    - Every time you make a call via 'rectifier.py', LiteLLM tracks the tokens.
    - Cost is calculated based on the model's pricing (e.g., GPT-4o, Llama 3).
    
    Commands:
    - Check Status:  python budget_checker.py
    - Show Guide:    python budget_checker.py --guide
    
    Programmatic Access:
    You can also check costs in your Python code:
    
    >>> response = completion(model="...", messages=[...])
    >>> cost = response._hidden_params.get('response_cost', 0)
    >>> print(f"Call cost: ${cost}")
    ===========================================
    """
    print(guide_text)


def get_key_info(api_key: str | None, base_url: str | None) -> dict[str, object]:
    """Fetch key information from the LiteLLM proxy.

    Args:
        api_key: Proxy key to inspect.
        base_url: LiteLLM-compatible proxy base URL.

    Returns:
        Provider key information response.
    """
    if not api_key:
        print("❌ Error: LLM_API_KEY not found in environment variables.")
        print("   Please check your .env file.")
        sys.exit(1)
    if not base_url:
        print("❌ Error: LLM_API_BASE not found in environment variables.")
        print("   Please check your .env file.")
        sys.exit(1)

    # Clean up base URL
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/key/info"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # The /key/info endpoint expects the key as a query param or in the header
        # We pass it as a query param 'key' as per standard LiteLLM Proxy docs
        response = requests.get(
            endpoint, headers=headers, params={"key": api_key}, timeout=10
        )
        response.raise_for_status()
        return cast(dict[str, object], response.json())
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: Could not connect to LiteLLM Proxy at {base_url}")
        print(f"   Details: {e}")
        sys.exit(1)


def display_budget(info: dict[str, object]) -> None:
    """Parse and display provider budget information.

    Args:
        info: Provider key information response.
    """
    # Extract fields safely
    wrapped = info.get("info")
    info_data = wrapped if isinstance(wrapped, dict) and wrapped else info

    maximum = info_data.get("max_budget")
    max_budget = float(maximum) if isinstance(maximum, (int, float)) else None
    spent = info_data.get("spend", 0.0)
    spend = float(spent) if isinstance(spent, (int, float)) else 0.0
    user_id = info_data.get("user_id", "Unknown User")

    # Calculate remaining
    # If max_budget is None, it usually means unlimited
    if max_budget is None:
        remaining = "Unlimited"
        status_icon = "infinity"
    else:
        remaining_val = max_budget - spend
        remaining = f"${remaining_val:.4f}"

        # Determine status icon
        usage_percent = (spend / max_budget) * 100 if max_budget > 0 else 0
        if usage_percent > 90:
            status_icon = "🔴"  # Critical
        elif usage_percent > 75:
            status_icon = "🟠"  # Warning
        else:
            status_icon = "🟢"  # Good

    print("\n📊 API Budget Status")
    print("-------------------")
    print(f"👤 User ID:    {user_id}")
    print(f"💸 Total Spend: ${spend:.4f}")

    if max_budget is not None:
        print(f"💰 Max Budget:  ${max_budget:.4f}")
        print(f"{status_icon} Remaining:   {remaining}")
    else:
        print("💰 Max Budget:  Unlimited")

    print("-------------------\n")


def main() -> None:
    """Parse arguments and display the guide or current provider balance."""
    parser = argparse.ArgumentParser(description="Check LiteLLM API Key Budget")
    parser.add_argument(
        "--guide", action="store_true", help="Show usage guide and exit"
    )
    args = parser.parse_args()

    if args.guide:
        print_guide()
        return

    key_hint = f"{API_KEY[:4]}...{API_KEY[-4:]}" if API_KEY else "not configured"
    print(f"Checking budget for key: {key_hint}")
    key_info = get_key_info(API_KEY, LITELLM_BASE_URL)
    display_budget(key_info)


if __name__ == "__main__":
    main()
