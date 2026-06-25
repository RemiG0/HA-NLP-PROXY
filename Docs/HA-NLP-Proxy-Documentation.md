# HA NLP Proxy - Dokumentacja i Instrukcja Obsługi

## Wprowadzenie
**HA NLP Proxy** to samodzielna usługa (proxy) kompatybilna z API OpenAI, zaprojektowana specjalnie do obsługi asystentów głosowych w Home Assistant. Jej głównym celem jest całkowicie lokalne przetwarzanie polskich poleceń głosowych za pomocą modeli uczenia maszynowego (`allegro/herbert-base-cased` + `LinearSVC`) bez konieczności odpytywania zewnętrznych chmur AI (OpenAI, Anthropic). Jeśli system nie jest pewny intencji użytkownika, zapytanie jest przekazywane (fallback) do wybranego, zewnętrznego LLM.

## Architektura i Komponenty Systemu
Aplikacja została zbudowana na języku Python 3.11 z wykorzystaniem frameworka **FastAPI**. Zarządzanie pakietami odbywa się przez `uv`.

- **`main.py`**: Główny punkt wejściowy aplikacji (FastAPI). Obsługuje endpoint proxy `/v1/chat/completions` naśladujący natywny format odpowiedzi OpenAI (w szczególności blok `tool_calls`).
- **`nlp.py`**: Moduł odpowiedzialny za ładowanie wyuczonych modeli Scikit-Learn oraz modelu językowego HuggingFace. Analizuje polecenia i dopasowuje intencje (np. `HassTurnOn`) i encje (np. `light.salon`).
- **`db.py`**: Baza danych SQLite (`ha_nlp.db`) zarządzana przez ORM `SQLModel`. Przechowuje konfigurację, logi inferencji, wyuczone encje oraz zdefiniowane przez użytkownika zdania treningowe.
- **`train.py`**: Skrypt uruchamiany asynchronicznie, którego celem jest przeliczenie embeddingów HerBERT i wyuczenie klasyfikatorów `LinearSVC` dla zebranych zdań i encji.
- **`ha_sync.py`**: Narzędzie synchronizujące dostępne w Home Assistant urządzenia (encje) z wewnętrzną bazą danych przez REST API. Zsynchronizowane urządzenia posłużą modelowi ML do klasyfikacji na co ma wpłynąć dana komenda.
- **Panel Admina (`routers/admin.py`, `templates/`)**: Webowy, renderowany po stronie serwera panel zarządzania wykorzystujący `Jinja2` i odświeżany asynchronicznie przez bibliotekę `HTMX`.

## Instalacja
Do uruchomienia tego środowiska na komputerze hosta niezbędny jest menedżer pakietów `uv`.

1. Klonowanie repozytorium i przejście do katalogu roboczego.
2. Pobranie zależności Pythona: `uv sync`
3. Uruchomienie proxy:
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```
*(Opcjonalnie środowisko może zostać uruchomione w kontenerze Docker w oparciu o przygotowany `Dockerfile`).*

## Konfiguracja Home Assistant

1. Wejdź w **Ustawienia** -> **Urządzenia oraz usługi**.
2. Wybierz opcję **Dodaj integrację** i wyszukaj **OpenAI Conversation**.
3. Przy dodawaniu integracji wybierz ustawienia z adresem własnego serwera:
   - **Base URL**: `http://<ADRES_IP_PROXY>:8000/v1`
   - **API Key**: Wpisz dowolny ciąg znaków (proxy nie waliduje go przy lokalnym działaniu).
4. Po pomyślnym dodaniu, wejdź w ustawienia Asystentów Głosowych (Voice Assistants) w Home Assistant i wskaż utworzoną integrację "OpenAI Conversation" jako tę główną do obsługi poleceń konwersacyjnych.

## Panel Zarządzania i Trening (Admin UI)
Serwis dostępny jest pod adresem: `http://<ADRES_IP_PROXY>:8000/admin/`

1. **Dashboard:** Daje szybki wgląd w stan aplikacji: liczbę wyuczonych poleceń, poprawność ładowania modelu oraz logi ostatnich komend i przewidzianych przez model intencji (np. po wypowiedzeniu do mikrofonu komendy, log natychmiast się zaktualizuje o wynik klasyfikacji).
2. **Konfiguracja:** Miejsce, gdzie ustawiany jest adres URL lokalnego Home Assistant (potrzebny do synchronizacji urządzeń) i długoterminowy token dostępu z HA. Należy tam również skonfigurować _Fallback LLM_ jeśli chcemy używać np. ChatGPT (API Key OpenAI) jako spadochronu ratunkowego, w sytuacji gdy lokalny model nie zrozumie polecenia.
3. **Trening Encji:** Tutaj należy pobrać encje (urządzenia) z Home Assistant klikając guzik synchronizacji. Można następnie w bazie podejrzeć które światła czy rolety zostały zaimportowane. Zostaną one wykorzystane do treningu by asystent lokalny wiedział do jakich urządzeń chcemy się odwoływać.
4. **Trening Intencji:** Serce całego rozwiązania. Należy tu dodawać w języku polskim zdania wywołujące konkretną intencję Home Assistanta. 
   - *Przykład:* Intencja `HassTurnOn`, Zdanie: `Włącz urządzenie`.
   - *Przykład:* Intencja `HassTurnOff`, Zdanie: `Proszę wyłącz światło`.

**Uwaga:** Zawsze po pobraniu nowych encji z Home Assistant lub modyfikacji zdań intencji należy w panelu kliknąć **"Przetrenuj modele"** (lub `🔁 Retrain Models`). Uruchomi to skrypt ML w tle, który po minucie zastąpi wygenerowane pliki `.joblib` z wyuczonymi modelami.

## System Odrzucania / Fallback LLM
Aplikacja mierzy własną pewność sklasyfikowanego polecenia głosowego (tzw. _Confidence Score_). Jeśli po przetworzeniu w lokalnym modelu `LinearSVC` wynik (Score) nie osiągnie progu ustalonego w zakładce **Konfiguracja** (domyślnie `0.6`), system przekaże pełne żądanie do zapasowego LLM.

- **Dla zewnętrznych systemów LLM (np. ChatGPT, Azure)**: Konieczne jest podanie adresu do API OpenAI oraz klucza w zakładce Konfiguracji.
- **Dla własnych modeli generatywnych (Ollama, LM Studio)**: Można tam wpisać adres działającego lokalnie na innym porcie mocarnego modelu AI bez podawania klucza API, dzięki czemu 100% zapytań (zarówno te przetworzone lokalnie przez HA NLP Proxy jak i te odesłane po obniżonej pewności do Ollamy) wciąż nigdy nie wyjdą poza infrastrukturę domową.

## Testy Automatyczne
Aplikacja posiada rozbudowany pakiet testów automatycznych stworzony w oparciu o bibliotekę `pytest`, który dba o stabilność logiki, komunikacji i działania proxy.

### Jak uruchomić testy?
Aby uruchomić wszystkie testy wraz z raportem pokrycia kodu (Test Coverage), wejdź do głównego katalogu aplikacji i wywołaj polecenie:

```bash
cd ha-nlp-proxy
uv run pytest tests/ -v --cov=.
```

### Co jest testowane?
Środowisko testowe omija konieczność ładowania potężnych modeli językowych (tzw. "mockowanie" za pomocą biblioteki `unittest.mock`), dzięki czemu testy wywołują się błyskawicznie bez użycia dużej ilości pamięci RAM lub karty graficznej:

1. **Baza Danych (`tests/test_db.py`)** 
   - Weryfikacja odczytów i zapisów konfiguracji przy pomocy dedykowanej bazy działającej tylko w pamięci (`:memory:`).
   - Testowanie zapisu logów komunikacji.
2. **Logika NLP (`tests/test_nlp.py`)** 
   - Testowanie poprawności progów odcięcia (Confidence Thresholds). Sprawdza, czy przy wysokiej przewidywanej pewności model zwróci odpowiednią intencję, a przy zbyt niskiej – prawidłowo zwróci `None` by przekierować ruch na Fallback LLM.
3. **API i Fallback LLM (`tests/test_main.py`)** 
   - Weryfikacja cyklu zapytania i odpowiedzi, potwierdzająca że format `tool_calls` dla Home Assistanta zachowuje poprawną formę kompatybilną z OpenAI.
   - Symulacje żądań, które odpytują zapasowe API (np. zewnętrzne modele).
4. **Panel Webowy Admina (`tests/test_admin.py`)**
   - Zautomatyzowane żądania do serwera potwierdzające, że wszystkie szablony `.html` ładują się pomyślnie, formularze wysyłają właściwe dane i odświeżają stany, zapobiegając błędom po refaktoryzacji kodu.
