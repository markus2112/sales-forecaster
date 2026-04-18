from fastapi import HTTPException
from datetime import date


def validate_sales_data(data):

    # ---------------- DATE ----------------
    if data.date > date.today():
        raise HTTPException(
            status_code=400,
            detail="Future date not allowed"
        )

    # ---------------- LOGICAL CHECK ----------------
    if data.stock == 0 and data.sales > 0:
        raise HTTPException(
            status_code=400,
            detail="Sales cannot occur when stock is 0"
        )

    return True