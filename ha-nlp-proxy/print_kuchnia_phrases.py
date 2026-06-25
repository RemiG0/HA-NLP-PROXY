import db
from sqlmodel import Session, select

def get_locative_variants(base_name):
    variants = {base_name}
    if base_name.endswith('nia'): variants.add(base_name[:-3] + 'ni')
    if base_name.endswith('ka'): variants.add(base_name[:-2] + 'ce')
    if base_name.endswith('on'): variants.add(base_name + 'ie')
    if base_name.endswith('aż'): variants.add(base_name + 'u')
    if base_name.endswith('ód'): variants.add(base_name[:-2] + 'odzie')
    if base_name.endswith('ro'): variants.add(base_name[:-2] + 'rze')
    if base_name.endswith('ój'): variants.add(base_name[:-2] + 'oju')
    if base_name.endswith('rz'): variants.add(base_name + 'u')
    return variants

with Session(db.engine) as s:
    e = s.exec(select(db.Entity).where(db.Entity.friendly_name == 'Kuchnia')).first()
    
    base_names = [e.friendly_name]
    smart_phrases = set()
    for name in base_names:
        name_lower = name.lower()
        clean_name = name_lower.replace(" do ", " ").replace(" w ", " ")
        name_variants = get_locative_variants(name_lower)
        clean_variants = get_locative_variants(clean_name)
        
        for prefix in ["", "włącz ", "wyłącz ", "ustaw ", "zgaś ", "zapal ", "otwórz ", "zamknij "]:
            smart_phrases.add(f"{prefix}{name}")
            smart_phrases.add(f"{prefix}{clean_name}")

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

    print("\n".join(sorted(list(smart_phrases))))
