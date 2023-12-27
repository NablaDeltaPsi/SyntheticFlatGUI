Python-GUI zum Erstellen von synthetischen Flats und weiterer Rohbild-Analysen

Für Windows-Benutzer ist unter Releases eine ZIP-Datei mit kompiliertem Programm verfügbar. Zum Starten nach Herunterladen und Entzippen "SyntheticFlatGUI.exe" ausführen.

![grafik](https://github.com/NablaDeltaPsi/syntheticflatgui/assets/98178269/38d00116-180a-4997-882a-1b5b1fd106da)
 
Um in der Astrofotografie die Vignettierung von Objektiven zu Korrigieren, muss vor dem Stacken ein (Master-) Flat-Frame von den Einzelbildern dividiert werden. Normalerweise erstellt man das Master-Flat, indem mehrere Bilder einer weißen Wand aufgenommen und kombiniert werden, um möglichst wenig zusätzliches Rauschen ins Bild zu bringen.

Dennoch bringen qualitativ minderwertige oder unpassende Flats oft auch Probleme wie Raining Noise, Überkorrektur oder ringartige Strukturen ins Bild. Dieses Programm ermöglicht die einfache Erstellung synthetischer Flats, um diese Probleme zu umgehen.

### Schaltflächen
- Load files: Auswählen eines oder mehrerer RAW-Bilder.
- Set bias value: Abziehen eines Kameraabhängigen Offset-Wertes vor Berechnung aller folgenden Funktionen
- Start: Anwenden der Funktionen unter "Options" 

### Options
Ein/Aus Schalter der Hauptfunktionen
- Correct gradient: Ein Gradient im Bild muss vor der Erstellung eines Flats korrigiert werden.
- Nearest neighbor pixelmap: Diagramm zum Testen auf den Star-eater-Algorithmus (https://www.change.org/p/sony-remove-star-eater-en, https://www.cloudynights.com/topic/635441-aa-filter-spatial-filter-and-star-colours/page-2#entry8914661)
- Calculate histogram: Selbsterklärend, speichert eine CSV mit dem Histogramm des Rohbilds (RGB) ab.
- Calculate radial profile: Ermittelt die Helligkeitskurve des Bildes in Abhängigkeit zum Abstand zur Bildmitte und speichert CSVs dazu ab. Zur Ermittlungslogik siehe "Statistics".
- Export synthetic flat: Mit dem berechneten radialen Profil (Option wird automatisch gechecked) wird ein normiertes Flat (16-bit TIF) berechnet, das gleich groß ist wie das Ursprungsbild.

### Settings
Kleinere Schalter zum Beeinflussen der Hauptfunktionen
- Write pickle file: Schreiben einer PKL Datei des gedebayerten Bildes zum schnelleren Ausführen darauffolgender Läufe
- Histogram of largest circle: Das Histogramm eines Flats wird normalerweise durch das rechteckige Beschneiden beeinflusst. Mit dieser Option wird das Histogramm nur für den größtmöglichen Kreis im Bild berechnet.
-  Extrapolate inside max: Radiale Profile können ihr Maximum statt bei Radius 0 bei größeren Radii aufweisen, wodurch im synthetischen Flat Ringe entstehen würden. Mit dieser Option wird das radiale Profil beim Maximum abgeschnitten und zum Zentrum hin mit einer quadratischen Funktion extrapoliert.
-  Export synthetic flat as grey: Für das synthetische Flat wird für alle drei Farbkanäle das gleiche gemittelte radiale Profil verwendet.
-  Export synthetic flat debayered: Für das synthetische Flat wird RGB in drei Ebenen statt mit RGGB-Muster in einer Ebene geschrieben.
-  Scale synthetic flat like original: Die relative Intensität des synthetischen Flats wird nicht auf 1 normiert sondern dem Originalbild entsprechend angepasst.



