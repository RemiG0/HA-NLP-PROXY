import nlp
# Print what HA sent us in the seed
import db
from sqlmodel import Session, select
import json

with Session(db.engine) as s:
    sample = s.exec(select(db.IntentSample).where(db.IntentSample.intent == "HassTurnOn").limit(5)).all()
    for row in sample:
        print(row.sentence)
