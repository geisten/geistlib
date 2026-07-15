# Ein Sprachassistent, der auf deinem Raspberry Pi lebt — und nirgendwo sonst

1,2 GB. Ein Raspberry Pi 5. Kein Netzwerk-Hop, kein Account, keine Cloud. Du
sagst „Schalte das Licht im Flur ein", und rund zwei Sekunden später ist es an —
und der einzige Computer, der dich gehört hat, ist der auf deinem Regal.

Das ist geist-home: eine einzelne Binärdatei, die die ganze Schleife auf der
Platine ausführt — Sprachverständnis, Werkzeug-Routing, Gerätesteuerung. Das
Modell ist in die ausführbare Datei eingebacken; das Einzige, womit sie je
spricht, ist deine Home-Assistant-Instanz im lokalen Netz und die Lampe, nach
der du gefragt hast. Die meisten Sprachassistenten schicken deine Worte ins
Rechenzentrum von jemand anderem und hoffen, dass es den Dienst nächstes Jahr
noch gibt. Dieser hier setzt auf das Gegenteil.

So funktioniert es — und die drei Ideen, die ein 2-Bit-Sprachmodell auf einer
80-Euro-Platine reaktionsschnell genug machen, um es tatsächlich zu benutzen.

## Idee 1: Das Modell routet, es steuert nie

Der Reflex bei LLM-Agenten ist, das Modell alles machen zu lassen — die Anfrage
parsen, das Gerät wählen, die Aktion festlegen, den API-Aufruf erzeugen. So
bekommt man auch eine Lampe, die ausgeht, weil das Modell eine Entity-ID
halluziniert hat.

geist-home zieht eine harte Grenze. Das Modell tut genau eine
wahrscheinlichkeitsbehaftete Sache: Es wählt, **welches Werkzeug** die Anfrage
bearbeitet — ein *Kommando*-Werkzeug (ein/aus, dimmen, Temperatur setzen) oder
ein *Status*-Werkzeug (den Zustand eines Geräts lesen). Alles danach ist
deterministisches C:

- Welches Gerät? Gematcht gegen eine Registry-Datei
  (`entity | domain | Aliasse`), indem gezählt wird, wie viele Alias-Wörter in
  der Anfrage vorkommen. „Licht im Wohnzimmer" erzielt 2 Punkte auf
  `licht wohnzimmer` und schlägt das generische `licht` — kein Raten.
- Welche Aktion? Aus Verben und Zahlen in der Anfrage geparst.
- Ist es erlaubt? Schreibzugriffe sind auf Licht, Schalter, Klima und Rollläden
  beschränkt. Garagentore und Alarmanlagen werden grundsätzlich abgelehnt. Eine
  Tür zu entriegeln braucht einen zweistufigen Bestätigungsablauf — das Modell
  transportiert nur das Bestätigungswort, es *entscheidet* nie eine
  Sicherheitsfrage.

Das Sicherheitsversprechen ist nicht „wir haben das Modell per Prompt zum
Wohlverhalten überredet". Es ist, dass der Code zum Surfen im Web oder zum Lesen
deiner Dateien **gar nicht in der Binärdatei ist**. Ein `make home`-Build
kompiliert die zwei Home-Werkzeuge hinein und sonst nichts. Fester Umfang zur
Build-Zeit.

## Idee 2: Das Routing ist kalibriert, nicht aus dem Bauch

Frag ein kleines Modell „welches Werkzeug?", und seine rohe Präferenz ist durch
Token-Häufigkeit verzerrt — es mag Werkzeugnamen, die zufällig gängige Wörter
sind. Deshalb bewertet geist-home die Wahrscheinlichkeit des ersten Tokens jedes
Werkzeugnamens und **zieht dann eine Baseline ab**: denselben Wert für eine
inhaltsleere Anfrage. Was übrig bleibt (eine punktweise Transinformation), ist
das von der Anfrage getriebene Signal.

Darüber sitzen deterministische Evidenz-Gates. Eine Anfrage ohne Home-Wörter
kann nicht zu einem Home-Werkzeug routen. Eine Anfrage, die einen Raum nennt aber
kein konkretes Gerät („Wie ist der Status im Wohnzimmer?"), bekommt den *ganzen
Raum* zurückgelistet — `light.wohnzimmer: an, cover.living_room_window: offen` —,
denn ein Gerät ist „im" Raum, wenn seine Aliasse das Raumwort tragen.

Und eine Regel, die nichts kostet — kein Modell-Durchlauf: Die Lese/Schreib-Grenze
entscheidet die **Satzform**, nicht der Namens-Score. Ein Imperativ („Dimme das
Licht") meint das Kommando-Werkzeug; eine Frage („Wie warm ist es?") den Status.
Die Erst-Token-Wahrscheinlichkeiten verwechseln die beiden ständig; die Grammatik
des Satzes tut es nicht.

## Idee 3: Latenz ist ein Prefill-Problem — also hör auf, sie neu zu bezahlen

Hier wurde es interessant. Die ersten Messungen auf dem Pi zeigten einen Turn von
12–13 Sekunden. Inakzeptabel für „mach das Licht an". Der offensichtliche
Verdächtige — ein 2-Bit-Modell, das langsam dekodiert — stellte sich als falsch
heraus. Home-Antworten sind kurz und meist deterministisch (`OK: light.flur →
an`); das Modell dekodiert kaum etwas. **Die gesamten Kosten sind Prefill** — das
Durchschieben des Prompts durch das Netz vor dem ersten Ausgabe-Token.

Drei Fixes, jeder gemessen:

1. **Pre-Warm.** Die 12–13 s waren der *kalte erste Turn* — das Primen des
   KV-Caches und das Berechnen der Routing-Baseline. Also lässt der Daemon beim
   Start einen Wegwerf-Turn laufen, bevor er Anfragen annimmt. Die erste *echte*
   Anfrage fällt auf ~3 s.

2. **Ein gepinntes Routing-Menü.** Jeder Turn schob das Werkzeug-Menü (~60
   Tokens) nur fürs Routing erneut durchs Modell. Dieses Menü ist konstant, also
   pinnen wir es einmal in den KV-Cache und spulen jeden Turn dorthin zurück —
   und schieben nur die ~8 Tokens der eigentlichen Anfrage nach. Routing-Prefill:
   2,0 s → 0,45 s. (Der subtile Teil: dazu musste der Prompt so umgeordnet
   werden, dass das konstante Menü ein Präfix ist und die Anfrage ein Suffix —
   und dann per Eval bewiesen werden, dass die Umordnung *null*
   Routing-Entscheidungen verschob.)

3. **Ein begrenztes Kontextfenster.** Der Daemon hält eine lange Konversation,
   damit Nachfragen funktionieren („mach das Licht an" → „dimme es herunter" löst
   *es* auf). Aber ein unbegrenztes Transcript ließ den Prefill pro Turn über
   einen Tag Nutzung Richtung *Minuten* klettern. Das Fenster auf die letzten
   ein, zwei Turns zu deckeln hält ihn flach — und der Nachfrage-Kontext, den man
   wirklich braucht, liegt ohnehin nur einen Turn oder zwei zurück.

Das Ergebnis: Ein warmer Turn pendelte sich bei **~2 Sekunden ein und bleibt
dort**, herunter von „~4 s und steigend". Die Schlagzeilen-Zahl ging nie um die
Geschwindigkeit des Modells. Sie ging darum, nicht dieselben Tokens in jedem Turn
erneut zu bezahlen.

## Warum diese Form die richtige ist

Der Reflex im Jahr 2026 ist, ein größeres Modell auf das Problem zu werfen. Aber
eine Home-Appliance braucht kein Modell, das alles kann — sie braucht eines, das
zuverlässig zwischen „Kommando" und „Status" wählt und einen Gerätenamen
heraushebt. Das ist eine Aufgabe, die ein 2-Bit-2B-Modell gut erledigt, *wenn es
in deterministisches Gerüst gewickelt ist, das alles übernimmt, dem man es nicht
anvertrauen sollte.*

Kleines Modell, enger Umfang, alles gemessen, nichts verlässt das Haus. Das
Interessante war nie, das Modell größer zu machen. Es war, es für weniger
zuständig zu machen.

---

*geist-home ist ein `make home`-Build von geist, einer von Grund auf gebauten
CPU-Inference-Engine in C. Deployment-Anleitung: [HOME.md](../HOME.md).*
