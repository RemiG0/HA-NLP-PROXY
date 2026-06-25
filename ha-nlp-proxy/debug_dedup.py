import db
import train
from sqlmodel import Session, select
from collections import defaultdict

with Session(db.engine) as s:
    entities = s.exec(select(db.Entity).where(db.Entity.enabled == True)).all()

entity_phrases = defaultdict(set)
for e in entities:
    base_names = [e.friendly_name]
    if hasattr(e, 'aliases') and e.aliases:
        for alias in e.aliases.split(','):
            alias = alias.strip()
            if alias:
                base_names.append(alias)
    
    smart_phrases = set()
    for name in base_names:
        name_lower = name.lower()
        clean_name = name_lower.replace(" do ", " ").replace(" w ", " ")
        name_variants = train.get_locative_variants(name_lower)
        clean_variants = train.get_locative_variants(clean_name)
        
        for prefix in ["", "włącz ", "wyłącz ", "ustaw ", "zgaś ", "zapal ", "otwórz ", "zamknij "]:
            smart_phrases.add(f"{prefix}{name}")
            smart_phrases.add(f"{prefix}{clean_name}")
        
        if e.domain in ["light", "switch"] and "światł" not in name_lower:
            for v in clean_variants:
                smart_phrases.add(f"światło {v}")
                smart_phrases.add(f"włącz światło {v}")
                smart_phrases.add(f"zapal światło {v}")
                smart_phrases.add(f"zgaś światło {v}")
        elif e.domain == "cover" and "rolet" not in name_lower:
            for v in clean_variants:
                smart_phrases.add(f"rolety {v}")
                smart_phrases.add(f"zasłoń rolety {v}")
                smart_phrases.add(f"zamknij rolety {v}")
                smart_phrases.add(f"otwórz rolety {v}")
        elif e.domain == "vacuum":
            for v in clean_variants:
                smart_phrases.add(f"odkurzacz {v}")
                smart_phrases.add(f"odkurz {v}")
                smart_phrases.add(f"posprzątaj {v}")
        elif e.entity_id.startswith("area."):
            for v in name_variants:
                smart_phrases.add(f"światło w {v}")
                smart_phrases.add(f"włącz światło w {v}")
                smart_phrases.add(f"zapal światło w {v}")
                smart_phrases.add(f"zgaś światło w {v}")
                smart_phrases.add(f"wszystkie światła w {v}")
                smart_phrases.add(f"włącz wszystkie światła w {v}")
                smart_phrases.add(f"zgaś wszystkie światła w {v}")
                smart_phrases.add(f"rolety w {v}")
                smart_phrases.add(f"zasłoń rolety w {v}")
                smart_phrases.add(f"zamknij rolety w {v}")
                smart_phrases.add(f"otwórz rolety w {v}")
                smart_phrases.add(f"wszystkie rolety w {v}")
                smart_phrases.add(f"odkurz {v}")
                smart_phrases.add(f"posprzątaj w {v}")
                
    entity_phrases[e.entity_id].update(smart_phrases)

phrase_to_eids = defaultdict(list)
for eid, phrases in entity_phrases.items():
    for p in phrases:
        phrase_to_eids[p].append(eid)

final_phrases = defaultdict(list)
for eid, phrases in entity_phrases.items():
    unique_phrases = set()
    for p in phrases:
        eids = phrase_to_eids[p]
        if len(eids) == 1:
            unique_phrases.add(p)
        else:
            is_area = eid.startswith("area.")
            has_area = any(id.startswith("area.") for id in eids)
            if not is_area and has_area:
                continue
            elif is_area:
                areas = [id for id in eids if id.startswith("area.")]
                if len(areas) == 1:
                    unique_phrases.add(p)
            else:
                specifics = [id for id in eids if not id.startswith("area.")]
                if len(specifics) == 1:
                    unique_phrases.add(p)
    final_phrases[eid] = sorted(list(unique_phrases))

print("=== light.kitchen_light ===")
for p in final_phrases['light.kitchen_light']: print(p)

print("\n=== area.afdcbd6db1e5477f937193f4561e7e36 (Kuchnia) ===")
for p in final_phrases['area.afdcbd6db1e5477f937193f4561e7e36']: print(p)

