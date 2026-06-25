import db
from sqlmodel import Session, select
import json

with Session(db.engine) as s:
    e = s.exec(select(db.Entity).where(db.Entity.entity_id == 'light.kitchen_light')).first()
    print("friendly_name:", e.friendly_name)
    print("aliases:", e.aliases)
