from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///data.db", connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Message(Base):
    """
    SQLAlchemy model for storing analyzed messages.
    """
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    emotion = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    """
    Dependency for FastAPI to get a DB session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()