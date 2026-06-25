from sqlmodel import Session, select
from db import engine, IntentSample

SEED_DATA = [
    # HassTurnOn
    ("włącz", "HassTurnOn"),
    ("zapal", "HassTurnOn"),
    ("uruchom", "HassTurnOn"),
    ("odpal", "HassTurnOn"),
    ("zaświeć", "HassTurnOn"),
    ("odpal światło", "HassTurnOn"),
    ("niech stanie się jasność", "HassTurnOn"),
    ("włącz urządzenie", "HassTurnOn"),

    # HassTurnOff
    ("wyłącz", "HassTurnOff"),
    ("zgaś", "HassTurnOff"),
    ("zgas", "HassTurnOff"),
    ("zatrzymaj", "HassTurnOff"),
    ("wyłącz urządzenie", "HassTurnOff"),
    ("ciemność", "HassTurnOff"),
    
    # HassToggle
    ("przełącz", "HassToggle"),
    ("zmień stan", "HassToggle"),
    ("przerzuć", "HassToggle"),

    # HassOpenCover
    ("otwórz", "HassOpenCover"),
    ("podnieś", "HassOpenCover"),
    ("odsłoń", "HassOpenCover"),
    ("otwórz rolety", "HassOpenCover"),
    ("otwórz żaluzje", "HassOpenCover"),
    ("do góry", "HassOpenCover"),
    ("otwórz okno", "HassOpenCover"),

    # HassCloseCover
    ("zamknij", "HassCloseCover"),
    ("opuść", "HassCloseCover"),
    ("zasłoń", "HassCloseCover"),
    ("zamknij rolety", "HassCloseCover"),
    ("zamknij żaluzje", "HassCloseCover"),
    ("na dół", "HassCloseCover"),
    ("zamknij okno", "HassCloseCover"),

    # HassVacuumStart (or TurnOn depending on how vacuum is exposed, usually HA intercepts Vacuum commands)
    ("odkurz", "HassVacuumStart"),
    ("posprzątaj", "HassVacuumStart"),
    ("rozpocznij odkurzanie", "HassVacuumStart"),
    ("wyślij robota", "HassVacuumStart"),
    ("odkurzacz start", "HassVacuumStart"),
    
    # HassVacuumReturnToBase
    ("wróć do bazy", "HassVacuumReturnToBase"),
    ("na stację", "HassVacuumReturnToBase"),
    ("odkurzacz do bazy", "HassVacuumReturnToBase"),
    ("skończ sprzątać", "HassVacuumReturnToBase"),
]

def seed():
    with Session(engine) as session:
        existing = session.exec(select(IntentSample)).all()
        existing_sentences = {e.sentence.lower() for e in existing}
        
        added = 0
        for sentence, intent in SEED_DATA:
            if sentence.lower() not in existing_sentences:
                session.add(IntentSample(sentence=sentence, intent=intent))
                added += 1
                
        session.commit()
        print(f"Dodano {added} nowych próbek intencji do bazy.")

if __name__ == "__main__":
    seed()
