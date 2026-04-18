from pydantic import BaseModel, Field
from datetime import date


class SalesData(BaseModel):
    date: date
    sales: float = Field(..., gt=0)
    promotion: bool
    stock: int = Field(..., ge=0)
    holiday: bool