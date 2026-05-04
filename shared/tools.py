MOCK_USERS = {
    "user_123": {
        "name": "Alice Smith",
        "kyc_status": "Verified",
        "tier": "Premium",
        "balances": {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0}
    },
    "user_456": {
        "name": "Bob Jones",
        "kyc_status": "Pending",
        "tier": "Basic",
        "balances": {"USDT": 500.0, "BTC": 0.01, "ETH": 0.1}
    }
}

CURRENT_USER_ID = "user_123"


def check_kyc_status(user_id: str = None) -> dict:
    """Checks KYC compliance status for the current user."""
    uid = user_id or CURRENT_USER_ID
    user = MOCK_USERS.get(uid)
    if not user:
        return {"verified": False, "message": "User not found"}
    verified = user["kyc_status"] == "Verified"
    return {
        "verified": verified,
        "kyc_status": user["kyc_status"],
        "user_name": user["name"],
        "message": "" if verified else f"⚠️ KYC Compliance Warning: Your account ({user['name']}) has KYC status '{user['kyc_status']}'. Please complete verification for full access.",
    }


def get_user_profile(user_id: str = "user_123") -> dict:
    """Retrieves account balances, KYC status, and membership tier."""
    return MOCK_USERS.get(user_id, {"error": "User not found"})


def execute_spot_trade(symbol: str, side: str, amount: float) -> str:
    """Executes a spot trade on the exchange market."""
    price = 95000.0 if "BTC" in symbol else 3500.0
    cost = amount * price
    return f"Executed {side} of {amount} {symbol}. Estimated cost: {cost} USDT."


def execute_derivative_trade(symbol: str, side: str, amount: float, leverage: int) -> str:
    """Executes a leveraged futures position on the derivatives platform."""
    return f"Successfully opened a {side} position on {symbol} for {amount} units at {leverage}x leverage."


def execute_transfer(asset: str, amount: float, destination: str) -> str:
    """Transfers cryptocurrency assets to an external destination address."""
    return f"Transfer transaction confirmed: Sent {amount} {asset} to {destination}."
