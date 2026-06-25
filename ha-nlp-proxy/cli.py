import argparse
import asyncio
import sys

def run_sync():
    print("Rozpoczynam synchronizację encji z Home Assistant...")
    from ha_sync import sync_entities_from_ha
    try:
        asyncio.run(sync_entities_from_ha())
        print("✅ Synchronizacja zakończona sukcesem.")
    except Exception as e:
        print(f"❌ Błąd podczas synchronizacji: {e}")

def run_train():
    print("Rozpoczynam trening modeli (może to zająć chwilę)...")
    from train import train
    try:
        train()
        print("✅ Trening zakończony sukcesem.")
    except Exception as e:
        print(f"❌ Błąd podczas trenowania: {e}")

def run_test(text: str):
    print(f"Testowanie zdania: '{text}'")
    from nlp import load_models, classify
    from db import get_config
    
    try:
        threshold_str = get_config("confidence_threshold", "60.0")
        threshold = float(threshold_str)
        if threshold > 1.0:
            threshold = threshold / 100.0
    except Exception:
        threshold = 0.60
        
    print("Ładowanie modeli AI...")
    try:
        load_models()
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modeli (czy na pewno uruchomiłeś najpierw 'train'?): {e}")
        return
        
    intent, entity, i_score, e_score = classify(text, threshold=threshold, return_all=True)
    
    print("\n" + "="*40)
    print("🎯 WYNIKI ANALIZY")
    print("="*40)
    
    print(f"WYKRYTA INTENCJA:  {intent if intent else 'NIE ROZPOZNANO'}")
    print(f"Pewność intencji:  {i_score*100:.1f}% (Próg: {threshold*100:.1f}%)")
    print("-" * 40)
    entity_name_str = ""
    if entity:
        from db import engine, Entity
        from sqlmodel import Session, select
        with Session(engine) as session:
            db_ent = session.exec(select(Entity).where(Entity.entity_id == entity)).first()
            if db_ent:
                entity_name_str = f" ({db_ent.friendly_name})"

    print(f"WYKRYTA ENCJA:     {entity if entity else 'NIE ROZPOZNANO'}{entity_name_str}")
    print(f"Pewność encji:     {e_score*100:.1f}% (Próg: {threshold*100:.1f}%)")
    print("="*40)
    
    success = (i_score >= threshold) and (e_score >= threshold)
    
    if success:
        print("\n✅ SUKCES")
        from nlp import build_ha_arguments
        args_list = build_ha_arguments(entity, text)
        print("To polecenie zostałoby poprawnie zinterpretowane i wykonane w Home Assistant jako:")
        for arg_json in args_list:
            print(f'{{\n  "type": "function",\n  "function": {{\n    "name": "{intent}",\n    "arguments": {repr(arg_json)}\n  }}\n}}')
    else:
        print("\n⚠️ ODRZUCONE")
        print(f"Serwer HA-NLP zignorowałby to polecenie z powodu zbyt niskiej pewności lub braku rozpoznania (poniżej {threshold*100:.1f}%).")

def main():
    parser = argparse.ArgumentParser(description="HA-NLP Proxy CLI - Narzędzia diagnostyczne")
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")
    
    # Sync
    sync_parser = subparsers.add_parser("sync", help="Pobierz encje i obszary z Home Assistant (wszystkie aktualnie wystawione)")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Uruchom trening modeli AI na obecnej bazie danych SQLite")
    
    # Test
    test_parser = subparsers.add_parser("test", help="Przetestuj dowolne zdanie, by sprawdzić pewność (confidence) klasyfikacji")
    test_parser.add_argument("text", type=str, help="Zdanie do przetestowania, np. 'włącz światło w kuchni'")
    
    args = parser.parse_args()
    
    if args.command == "sync":
        run_sync()
    elif args.command == "train":
        run_train()
    elif args.command == "test":
        run_test(args.text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
