# EDA: Cervical Cancer (Risk Factors)

**Ziel:** Identifikation von Faktoren, die zu einer positiven Biopsie (Gebärmutterhalskrebs) führen.

## Datengrundlage
- Quelle: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- 858 Zeilen, 36 Spalten → 'risk_factors_cervical_cancer.csv'

### Beschreibung der Spalten
- (int) Age
- (int) Number of sexual partners
- (int) First sexual intercourse (age)
- (int) Num of pregnancies
- (bool) Smokes
- (bool) Smokes (years)
- (bool) Smokes (packs/year)
- (bool) Hormonal Contraceptives - Hormonelle Verhütungsmittel
- (int) Hormonal Contraceptives (years)
- (bool) IUD - Intrauterine Device, umgangssprachlich Spirale
- (int) IUD (years)
- (bool) STDs
- (int) STDs (number)
- (bool) STDs:condylomatosis
- (bool) STDs:cervical condylomatosis
- (bool) STDs:vaginal condylomatosis
- (bool) STDs:vulvo-perineal condylomatosis
- (bool) STDs:syphilis
- (bool) STDs:pelvic inflammatory disease
- (bool) STDs:genital herpes
- (bool) STDs:molluscum contagiosum
- (bool) STDs:AIDS
- (bool) STDs:HIV
- (bool) STDs:Hepatitis B
- (bool) STDs:HPV
- (int) STDs: Number of diagnosis
- (int) STDs: Time since first diagnosis
- (int) STDs: Time since last diagnosis
- (bool) Dx:Cancer - Dx bedeutet Diagnose Krebs (allgemein)
- (bool) Dx:CIN - Zervixdysplasie Diagnose
- (bool) Dx:HPV - HPV-Infektion nachgewiesn
- (bool) Dx - allgemein Diagnose
- (bool) Hinselmann: target variable
- (bool) Schiller: target variable
- (bool) Cytology: target variable
- (bool) Biopsy: target variable




## Bereinigte Schritte
- Falsche dtypes der Spalten umgewandelt (v.a. objects → bool)
- Fehlende Werte zunächst mim NaNs ersetzt, dann mit MICE ersetzt.
- Mice wandelt alle dtypes in floats um → zurückumwandeln


### Analyse der fehlenden Werte
#### MCAR‑Analyse (Little’s Test)
Um zu prüfen, ob die fehlenden Werte vollständig zufällig (MCAR) auftreten, wurde Little’s MCAR‑Test durchgeführt.
Der Test ergab p‑Werte nahe 0, was bedeutet:
- Die Missingness ist nicht MCAR
- Es gibt systematische Unterschiede zwischen Fällen mit und ohne Missing Values
- Ein einfaches listenweises Löschen würde zu Bias und Informationsverlust führen

Damit ist klar:
MCAR kann ausgeschlossen werden

#### MAR‑Analyse (Missing At Random)
Da Age und Biopsy die einzigen Variablen ohne Missing Values sind, wurden sie als potenzielle Prädiktoren für Missingness verwendet.
Für jede Variable mit fehlenden Werten wurden zwei Tests durchgeführt:

- t‑Test: prüft, ob sich das Alter zwischen Missing‑ und Nicht‑Missing‑Gruppe unterscheidet
- Chi‑Quadrat‑Test: prüft, ob Missingness vom Biopsy‑Status abhängt


**Starke MAR‑Hinweise über Alter (Age_p < 0.001)**

Besonders bei folgenden Variablen zeigen sich hochsignifikante Altersunterschiede:

- Hormonal Contraceptives

- Hormonal Contraceptives (years)

- IUD

- IUD (years)

- Alle STD‑bezogenen Variablen (STDs, STDs:number, alle einzelnen STD‑Diagnosen)

Diese Variablen haben Age‑p‑Werte zwischen 0.000088 und 0.000158.

Interpretation:  
Das Alter beeinflusst klar, ob Personen diese Fragen beantworten.
Gerade Fragen zu Sexualverhalten, Verhütung und STDs sind sensibel – es ist plausibel, dass jüngere oder ältere Personen diese Angaben eher verweigern.

→ MAR über Alter ist sehr wahrscheinlich.


**MAR‑Hinweise über Biopsy (Biopsy_p < 0.05)**

Einige Variablen zeigen signifikante Zusammenhänge zwischen Missingness und Biopsy‑Status:

- Hormonal Contraceptives (p ≈ 0.0069)

- Hormonal Contraceptives (years) (p ≈ 0.0069)

- STDs: Time since first diagnosis (p ≈ 0.0026)

- STDs: Time since last diagnosis (p ≈ 0.0026)

Interpretation:  
Personen mit positivem Biopsy‑Status beantworten bestimmte Fragen systematisch häufiger oder seltener.
Auch hier deutet das auf MAR hin.

**Variablen ohne MAR‑Hinweis (p ≥ 0.05)**
Einige Variablen zeigen weder Alters‑ noch Biopsy‑Effekte:

- Number of sexual partners

- First sexual intercourse

- Num of pregnancies

- Smoking‑Variablen

Für diese Variablen gibt es keinen Hinweis auf MAR über Age oder Biopsy.
Sie könnten MCAR sein – oder MAR über andere, nicht getestete Variablen.



## Wichtigste Erkenntnisse
- Boxplots der Features mit Farbkodierung nach Biopsy zeigen, dass es keine klaren Entscheidungsgrenzen gibt.
- Biopsy hat kaum einen linearen Zusammenhang mit den Features
- RandomForest-Classifier ist nicht signifikant besser für Klassifizierung als ein KNN-Classifier.
- Bei der RFECV wurden 34, also fast alle Features ausgewählt. Das macht auch grundsätzlich so Sinn, da ein RandomForest sehr von vielen Features profitiert.
- Dimensionsreduktion mit PCA behält etwa 76% der Informationen.

### Erhöhtes Risiko für Cervical Cancer
- Diagnose Tests wie Schiller, Hinselmann und Citology (stark)
- Dx: CIN (stark)
- Kürzlich diagnostizierte STDs (mittel)
- Lange Nutzung von hormonellen Verhütungsmitteln (mittel)
- Eine HPV Diagnose (mittel aber Entscheidungsgrenze ausgeprägt)
- je mehr Schwangerschaften die Person schon hatte (schwach)
- je älter die Patientin ist (schwach)
- je mehr STDs man schon hatte (schwach)
- Spiralen zur Verhütung (IUD) (schwach)
- Rauchen (Anzahl Schachteln pro Jahr und Jahre) (schwach aber Entscheidungsgrenze ersichtlich)




## Technik
- Python 3.13.12
- Module in requirements.txt
- 'Project.ipynb' – vollständiger Analyse-Workflow
- main.py zum Ausführen von Streamlit
- ml.py für Logik

## Autor
Marzio Allemann