Python-GUI zum Erstellen von synthetischen Flats und weiterer Rohbild-Analysen

Für Windows-Benutzer ist unter Releases eine ZIP-Datei mit kompiliertem Programm verfügbar. Zum Starten nach Herunterladen und Entzippen "SyntheticFlatGUI.exe" ausführen.

![SyntheticFlatGUI](https://github.com/NablaDeltaPsi/syntheticflatgui/assets/98178269/183a62cc-bcb2-4297-97c6-7e549e74389b)

### Einleitung
 
Um in der Astrofotografie die Vignettierung von Objektiven zu Korrigieren, muss vor dem Stacken ein (Master-) Flat-Frame von den Einzelbildern dividiert werden. Normalerweise erstellt man das Master-Flat, indem mehrere Bilder einer weißen Wand aufgenommen und kombiniert werden, um möglichst wenig zusätzliches Rauschen ins Bild zu bringen.

Dennoch bringen qualitativ minderwertige oder unpassende Flats oft auch Probleme wie Raining Noise, Überkorrektur oder ringartige Strukturen ins Bild. Dieses Programm ermöglicht die einfache Erstellung synthetischer Flats, um diese Probleme zu umgehen.

### Vorgehen

![combined](https://github.com/NablaDeltaPsi/syntheticflatgui/assets/98178269/a86f590c-fc2e-45e7-8022-476037b45724)

Die synthetischen Flats können entweder aus üblichen manuellen Wand-Flats oder sogar aus den Lights berechnet werden, solange keine größeren flächigeren Strukturen zu sehen sind (Milchstraße). Zur Erstellung wird das Bild erst vom Bias/Offset-Wert befreit, der manuell angegeben oder aus einem dunklen Bias-Frame geladen werden kann. Anschließend wird ein radiales Profil berechnet, mit einer auswählbaren Sigma-Clipping Statistik manipuliert um Spitzen von Sternen zu entfernen, und anschließend weiter geglättet (Savitzki-Golay Filter). Ausgegeben wird ein TIF in der gleichen Größe wie das Original.

### Verwendung

Zur Verwendung sollte von den Lights ebenfalls erst der gleiche Bias-Wert abgezogen werden. Anschließend kann das Ergebnis durch das synthetische Flat dividiert werden, um ein korrigiertes Bild zu erhalten. In Siril sind bspw. die entscheidenden Kommandos zur Verarbeitung des Flats:

\# tif to fits  
*load masters/master_flat.tif  
save masters/master_flat*  

\# convert with synthetic bias and hotpixel list  
*calibrate light -bias="=500" -flat=../masters/master_flat -cc=bpm ../masters/master_bias_hotpixels.lst -cfa -equalize_cfa -debayer -prefix=cal_*

### Schaltflächen
- Load files  
Auswählen eines oder mehrerer RAW-Bilder.
- Set bias value  
Abziehen eines Kameraabhängigen Offset-Wertes vor Berechnung aller folgenden Funktionen
- Bias from file  
Auswählen eines dunklen Bias Frames zur Berechnung des Bias-Wertes. Zur Berechnung wird ein 2-Sigma-Clipping verwendet, cold- und hotpixel werden also ignoriert.
- Start  
Anwenden der Funktionen unter "Options" 

### Options
Ein/Aus Schalter der Hauptfunktionen
- Correct gradient  
Ein Gradient im Bild muss vor der Erstellung eines Flats korrigiert werden.
- Nearest neighbor pixelmap  
Diagramm zum Testen auf den Star-eater-Algorithmus (https://www.change.org/p/sony-remove-star-eater-en, https://www.cloudynights.com/topic/635441-aa-filter-spatial-filter-and-star-colours/page-2#entry8914661)
- Calculate histogram  
Selbsterklärend, speichert eine CSV mit dem Histogramm des Rohbilds (RGB) ab.
- Calculate radial profile  
Ermittelt die Helligkeitskurve des Bildes in Abhängigkeit zum Abstand zur Bildmitte und speichert CSVs dazu ab. Zur Ermittlungslogik siehe "Statistics".
- Export synthetic flat  
Mit dem berechneten radialen Profil (Option wird automatisch gechecked) wird ein normiertes Flat (16-bit TIF) berechnet, das gleich groß ist wie das Ursprungsbild.

### Settings
Kleinere Schalter zum Beeinflussen der Hauptfunktionen
- Write pickle file  
Schreiben einer PKL Datei des gedebayerten Bildes zum schnelleren Ausführen darauffolgender Läufe
- Histogram of largest circle  
Das Histogramm eines Flats wird normalerweise durch das rechteckige Beschneiden beeinflusst. Mit dieser Option wird das Histogramm nur für den größtmöglichen Kreis im Bild berechnet.
-  Extrapolate inside max  
Radiale Profile können ihr Maximum statt bei Radius 0 bei größeren Radii aufweisen, wodurch im synthetischen Flat Ringe entstehen würden. Mit dieser Option wird das radiale Profil beim Maximum abgeschnitten und zum Zentrum hin mit einer quadratischen Funktion extrapoliert.
-  Export synthetic flat as grey  
Für das synthetische Flat wird für alle drei Farbkanäle das gleiche gemittelte radiale Profil verwendet.
-  Export synthetic flat debayered  
Für das synthetische Flat wird RGB in drei Ebenen statt mit RGGB-Muster in einer Ebene geschrieben.
-  Scale synthetic flat like original  
Die relative Intensität des synthetischen Flats wird nicht auf 1 normiert sondern dem Originalbild entsprechend angepasst.

### Statistics
Auswählen der verwendeten Statistik zur Berechnung des radialen Profils. Auf einem Ring (vgl. Bild oben) liegen mehrere Pixel mit unterschiedlichen Werten. Aus diesen Werten pro Ring kann der Mittelwert, der Median, das Maximum, oder das Minimum genommen werden, oder sie werden erst mit einem Sigma-Clipping aussortiert (empfohlen) und erst dann der Mittelwert genommen.

### Shrink
Die Berechnung des radialen Profils für ein 24 MP Bild kann einige Zeit dauern, obwohl eine so große Auflösung gar nicht nötig ist. Deshalb kann mit dieser Option das Bild verkleinert und die Statistik mit diesem Verkleinerten Bild berechnet werden. Wichtig: Die Verkleinerung spielt nur für die statistischen Funktionen (Pixelmap, Histogramm, radiales Profil) eine Rolle, das ausgegebene synthetische Flat hat stets die gleiche Größe wie das Ursprungsbild! 


