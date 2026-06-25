# Home Assistant NLP (HA-NLP)

**HA-NLP** to projekt wprowadzający zaawansowane przetwarzanie języka naturalnego (NLP) do systemu Smart Home opartego na Home Assistant, ze szczególnym naciskiem na **język polski**. Składa się on z samodzielnego serwisu proxy (znajdującego się w podkatalogu `ha-nlp-proxy/`), który lokalnie klasyfikuje intencje i encje w czasie rzeczywistym przy wykorzystaniu modeli uczenia maszynowego.

## Główne informacje o projekcie

Projekt powstał, by uniezależnić sterowanie domem głosowo od usług chmurowych (takich jak chmura OpenAI, Google Assistant czy Alexa), a zarazem znacząco poprawić jakość i naturalność komunikacji w języku polskim w porównaniu do standardowych mechanizmów wbudowanych w Home Assistant.

Rozwiązanie integruje się z asystentem głosowym (Assist) w Home Assistant za pośrednictwem wbudowanej integracji **OpenAI Conversation**. Proxy podszywa się pod API OpenAI, co pozwala na bezproblemową i natywną integrację bez pisania niestandardowych wtyczek w systemie HA. 

Sercem klasyfikacji jest polski model językowy **HerBERT** (`allegro/herbert-base-cased`) oraz klasyfikatory oparte na wektorach wsparcia (`LinearSVC` z biblioteki scikit-learn). W przypadku niezrozumienia komendy, system pozwala na przekazanie (fallback) pytania do mocniejszego, zapasowego LLM (np. lokalnie działającej Ollamy lub płatnego API chmurowego).

## Przypadki użycia (Use Cases)

1. **Lokalne, szybkie komendy:**
   Gdy przebywasz w salonie, mówisz *"zapal tu światło"*. Zamiast wysyłać to zdanie do chmury, proxy w ułamku sekundy klasyfikuje intencję (`HassTurnOn`) i encję (`light.salon`), odsyłając odpowiedź bezpośrednio do HA.
2. **Ochrona prywatności (Privacy First):**
   Jeśli Twoje polecenia brzmią *"rozbrój alarm, PIN 1234"* lub *"otwórz bramę wjazdową"*, HA-NLP gwarantuje, że te dane nie wyjdą poza lokalną sieć domową.
3. **Zapasowe zapytania o wiedzę świata (Fallback LLM):**
   Gdy powiesz *"jaka jest dziś pogoda w Paryżu?"*, lokalny klasyfikator (LinearSVC) stwierdzi bardzo niską pewność, ponieważ nie był trenowany na takich komendach. Proxy płynnie prześle zapytanie dalej (np. do Ollama `llama3` działającej na Twoim serwerze) i po chwili otrzymasz odpowiedź asystenta.

## Dlaczego HA-NLP jest lepsze od standardowego rozpoznawania intencji (Regex) w Home Assistant?

### 1. Elastyczność na naturalną, luźną mowę
- **HA (Regex):** Wymaga wypowiadania komend w bardzo określony, sztywny sposób. Np. "Włącz światło w salonie". Jeśli powiesz *"weź no odpal te lampy w salonie"*, Regex tego nie zrozumie.
- **HA-NLP:** Model ML wyuczony na wektorach z osadzeń językowych (embeddings) rozumie semantykę i kontekst zdania. *"Odpal"*, *"zaświeć"*, *"włącz"* to wektorowo bliskie pojęcia, dzięki czemu naturalna mowa działa niemal w 100% poprawnie.

### 2. Radzenie sobie z fleksją (odmianą) języka polskiego
- **HA (Regex):** Polska gramatyka (odmiana przez przypadki) jest dla standardowego HA koszmarem. Trzeba pisać dziesiątki reguł uwzględniających np. *salonu, salonie, salonem*.
- **HA-NLP:** Model językowy **HerBERT** (od Allegro) został pretrenowany na ogromnych polskich korpusach tekstowych (w tym polskiej Wikipedii czy forach). Rozumie on polską gramatykę "z pudełka" bez konieczności definiowania reguł odmiany.

### 3. Brak barier przy dodawaniu nowych urządzeń (Encji)
- **HA (Regex):** Każde nowe urządzenie (np. *Oczyszczacz powietrza Xiaomi*) musi być dodawane do plików konfiguracyjnych YAML asystenta, ze wszystkimi swoimi możliwymi odmianami.
- **HA-NLP:** Serwis posiada wbudowaną opcję synchronizacji encji wprost z Home Assistant. Pobiera nazwy urządzeń i jednym kliknięciem ("Przetrenuj modele") asystent poznaje wszystkie Twoje nowe urządzenia w domu.

### 4. Inteligentny Fallback
- **HA (Regex):** Komenda nierozpoznana zwraca błąd *"Nie zrozumiałem polecenia"*.
- **HA-NLP:** Komenda nierozpoznana, z niską pewnością przydziału (Confidence Score < 0.6) jest odsyłana do pełnoprawnego, inteligentnego agenta konwersacyjnego, z którym możesz normalnie porozmawiać. 

---

## Struktura Katalogów

* `/ha-nlp-proxy/` - Główny kod źródłowy aplikacji (FastAPI, Scikit-Learn, UI).
* `/Docs/` - Skrzynka z dokumentacją techniczną i archiwalnymi dokumentami projektu (m.in. Instrukcja Obsługi).

Aby rozpocząć pracę i uruchomić serwer, zajrzyj do [dokumentacji proxy](./ha-nlp-proxy/README.md) lub [głównej instrukcji użytkownika](./Docs/HA-NLP-Proxy-Documentation.md).

---

## Podziękowania i Licencje Trzecich Stron (Acknowledgments & Third-Party Licenses)

Projekt wykorzystuje poniższe technologie i modele open-source. Dziękujemy ich twórcom za udostępnienie ich społeczności:

- **[HerBERT (allegro/herbert-base-cased)](https://huggingface.co/allegro/herbert-base-cased)** autorstwa Allegro (udostępniany na licencji CC BY 4.0 / Apache 2.0). Serce silnika wektoryzującego język polski w tym projekcie.
- **[FastAPI](https://fastapi.tiangolo.com/)** – szybki framework webowy (Licencja MIT).
- **[Transformers](https://huggingface.co/docs/transformers/index)** (Hugging Face) – obsługa modeli językowych (Licencja Apache 2.0).
- **[Scikit-learn](https://scikit-learn.org/)** – algorytmy klasyfikacyjne Machine Learning (Licencja BSD 3-Clause).
- **[PyTorch](https://pytorch.org/)** – biblioteka tensorowa wspierająca sieci neuronowe (Licencja BSD-style).
- **[HTMX](https://htmx.org/)** – dynamiczne interfejsy UI bez pisania JavaScriptu (Licencja BSD 2-Clause).
- **[SQLModel](https://sqlmodel.tiangolo.com/)** – obsługa bazy danych łącząca SQLAlchemy i Pydantic (Licencja MIT).

Aplikacja HA-NLP Proxy *nie* dystrybuuje kodu źródłowego powyższych bibliotek (są one pobierane na maszynie dewelopera lub serwerze za pomocą menedżera pakietów `uv`/`pip`), jednak z wdzięcznością uznaje ich ogromny wkład w ten projekt. Zgodnie ze standardami licencjonowania oprogramowania otwartego, odpowiednie zastrzeżenia licencyjne znajdują się w repozytoriach oryginalnych autorów.
