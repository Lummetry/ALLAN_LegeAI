# Proces antrenare get_targs

## Parametri de antrenare
Scriptul de antrenare se regaseste ori pe Desktop, ori in folderul “bat_scripts” din repository si are numele “train_tags.bat”. In cadrul rularii acestui script, **utilizatorul trebuie sa ia urmatoarele decizii**: 
1. Sursa datelor de antrenare:
    1. Din baza de date: se extrag documentele din baza de date, se preproceseaza si se salveaza in folderul “C:\allan\_cache/_data/” sub forma unor fisiere care incep cu “tags_vX” unde X este un numar (ex: fisierele pot incepe cu numele tags_v2).
    2. Dintr-un fisier .csv: fisierul trebuie sa contina o singura coloana pe care sa se regaseasca id-uri de documente care vor reprezenta documentele pe care se antreneaza modelul. In folderul “C:\allan” exista un fisier “ids.csv” cu rol de exemplu. 
    3. Dintr-un folder local, provenit dintr-o extragere anterioara. La fiecare rulare datele (cu tot cu preprocesarea necesara antrenarii) sunt salvate in folderul “C:\allan\_cache/_data” (vezi si primul punct). Acest mecanism permite reantrenarea unui model cu datele extrase anterior. Acest mecanism exista pentru ca procesul de extragere a datelor si preprocesare este unul de durata (poate dura si peste 6 ore).
2. Numarul de epoci de antrenare. **Se recomanda sa se foloseasca valoarea default** gasita in urma procesului de antrenare-validare-testare (5 pentru get_tags).
3. Numarul de exemple dintr-un batch (batch size): Poate fi crescut pentru a facilita o antrenare mai rapida. Momentan este setat la valoarea default, valoare maxima pentru configuratia actuala. **Se recomanda sa se foloseasca valoarea default**.
4. Dimensiunea contextului pentru modelul BERT (maxim 512): **Se recomanda sa se foloseasca valoarea default (512)**.
5. Numarul de predictii pentru evaluare: strict pentru evaluarea de la finalul scripturilor, este numarul de predictii pe care il luam in calcul. ATENTIE: modelul poate prezice oricate valori (argumentul “TOP_N” din API!), dar pentru primele predictii modelul este intotdeauna mai “sigur” pe predictiile facute. Pentru evaluare limitam numarul predictiilor la acest parametru pentru a verifica performanta modelului: cate din primele k predictii sunt relevante? Aceasta valoare nu influenteza in niciun fel modelul final, el va putea fi apelat cu orice valoare “TOP_N”.

**Exceptand sursa datelor care depinde de tipul de rulare dorit, se recomanda sa fie folosite valorile default pentru toti ceilalti parametri**: epoci de antrenare, batch size, dimensiunea contextului pentru BERT. 

La finalul antrenarii urmeaza cateva informatii legate de numele sub care au fost salvate datele (daca e nevoie, vezi info la “sursa datelor”) si folderul in care a fost salvat modelul. In continuare urmeaza partea de evaluare. Pentru evaluare avem nevoie de un set de date extern (care nu apare in antrenare!). Pentru aceasta am folosit exemplele date in sheet-ul cu teste. Am constuit fisierul test_samples_tags.csv cu doua coloane: text si label-uri. Acestea se gasesc in “C:\allan”. Evaluarea finala se face pe aceste exemple.

In final exista optiunea de a face deploy la modelul tocmai antrenat, astfel el ajunge “live”. 

## Observatii
- Antrenarea modelelor se face pe ~170,000 de exemple pentru get_tags. Testele pentru ‘test_samples_tags.csv’ contin foarte putine exemple.
Chiar si acelasi model antrenat de doua ori pe aceleasi date, cu aceeasi parametri poate da rezultate diferite pe aceste teste.
Avand foarte putin exemple, o predictie diferita cantareste foarte putin in performanta finala.
**De aceea se recomanda extinderea setului de teste (practic extinderea fisierului test_samples_tags.csv) pana la cel putin 100 de exemple (atentie: exemple care nu se gasesc deja in setul de antrenare)**. Cu cat sunt mai multe si mai diverse testele cu atat performanta pe acestea este mai reprezentativa pentru performanta generala a modelului.
