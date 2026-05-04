from pydantic import BaseModel, Field


class ProfileInput(BaseModel):
    user_id: str = Field(default="user_123", description="The user ID to look up")

class ProfileOutput(BaseModel):
    name: str
    kyc_status: str
    membership_tier: str
    balances: dict


class SpotTradeInput(BaseModel):
    symbol: str = Field(description="The cryptocurrency pair symbol, e.g., 'BTC/USDT'")
    side: str = Field(description="Must be 'buy' or 'sell'")
    amount: float = Field(description="The amount of cryptocurrency to trade")

class SpotTradeOutput(BaseModel):
    execution_receipt: str
    estimated_cost: str


class DerivativeTradeInput(BaseModel):
    symbol: str = Field(description="The futures symbol, e.g., 'ETH-PERP'")
    side: str = Field(description="Must be 'long' or 'short'")
    amount: float = Field(description="Position size")
    leverage: int = Field(description="Leverage multiplier")

class DerivativeTradeOutput(BaseModel):
    confirmation: str


class TransferInput(BaseModel):
    asset: str = Field(description="Cryptocurrency asset symbol, e.g., 'USDT'")
    amount: float = Field(description="Amount to send")
    destination: str = Field(description="Blockchain address or wallet ID")

class TransferOutput(BaseModel):
    transaction_receipt: str
