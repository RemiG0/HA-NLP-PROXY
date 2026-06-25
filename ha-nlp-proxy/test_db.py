import db
from sqlmodel import text, Session

try:
    with Session(db.engine) as s:
        s.exec(text("ALTER TABLE entity ADD COLUMN area_id VARCHAR"))
        s.commit()
        print("Added area_id column")
except Exception as e:
    print("Already exists or error:", e)
