import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- DATABASE URL ----------------
DATABASE_URL = "postgresql://sales_user:oys0d3rLRRrpLM1qt8iqcmqr6LqdEUQl@dpg-d7m7c94vikkc73fq0ksg-a.oregon-postgres.render.com/sales_forecaster_database"

# Render compatibility fix
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )

# ---------------- ENGINE ----------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False
)

# ---------------- SESSION ----------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ---------------- DEPENDENCY ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()
