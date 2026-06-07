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
- (bool) Dx:HPV - HPV-Infektion nachgewiesen
- (bool) Dx - allgemein Diagnose
- (bool) Hinselmann: target variable
- (bool) Schiller: target variable
- (bool) Cytology: target variable
- (bool) Biopsy: target variable


## Bereinigte Schritte
- Falsche dtypes der Spalten umgewandelt (v.a. objects → bool)
- Fehlende Werte zunächst mit NaNs ersetzt, dann mit MICE ersetzt.
- MICE wandelt alle dtypes in floats um → zurückumwandeln


### Analyse der fehlenden Werte

#### Untersuchung auf MAR (Missing At Random)
Da Age und Biopsy die einzigen Variablen ohne Missing Values sind, wurden sie als potenzielle Prädiktoren für Missingness verwendet.
Für jede Variable mit fehlenden Werten wurden zwei Tests durchgeführt:

- **t-Test**: Prüft, ob sich das Alter zwischen Missing- und Nicht-Missing-Gruppe unterscheidet
- **Chi-Quadrat-Test**: Prüft, ob Missingness vom Biopsy-Status abhängt

**Starke MAR-Hinweise über Alter (p < 0.005)**

Folgende Variablen zeigen hochsignifikante Altersunterschiede:

- Hormonal Contraceptives (p = 0.0001)
- Hormonal Contraceptives (years) (p = 0.0001)
- IUD (p = 0.0015)
- IUD (years) (p = 0.0015)
- Alle STD-bezogenen Variablen (p = 0.0002)
  - STDs, STDs (number)
  - STDs:condylomatosis, STDs:cervical condylomatosis
  - STDs:vaginal condylomatosis, STDs:vulvo-perineal condylomatosis
  - STDs:syphilis, STDs:pelvic inflammatory disease
  - STDs:genital herpes, STDs:molluscum contagiosum
  - STDs:AIDS, STDs:HIV, STDs:Hepatitis B, STDs:HPV

Interpretation:  
Das Alter beeinflusst klar, ob Personen diese Fragen beantworten.
Gerade Fragen zu Sexualverhalten, Verhütung und STDs sind sensibel – es ist plausibel, dass jüngere oder ältere Personen diese Angaben eher verweigern.

→ MAR über Alter ist sehr wahrscheinlich.

**MAR-Hinweise über Biopsy (p < 0.01)**

Einige Variablen zeigen signifikante Zusammenhänge zwischen Missingness und Biopsy-Status:

- STDs: Time since first diagnosis (p = 0.0026)
- STDs: Time since last diagnosis (p = 0.0026)
- Hormonal Contraceptives (p = 0.0070)
- Hormonal Contraceptives (years) (p = 0.0070)

Interpretation:  
Personen mit positivem Biopsy-Status beantworten bestimmte Fragen systematisch häufiger oder seltener.
Auch hier deutet das auf MAR hin.

**Variablen ohne MAR-Hinweis (p ≥ 0.05)**

Folgende Variablen zeigen weder Alters- noch Biopsy-Effekte:

- Number of sexual partners (p = 0.892)
- First sexual intercourse (p = 1.000)
- Num of pregnancies (p = 0.101)
- Smokes (p = 1.000)
- Smokes (years) (p = 1.000)
- Smokes (packs/year) (p = 1.000)

Für diese Variablen gibt es keinen Hinweis auf MAR über Age oder Biopsy.
Sie könnten MCAR sein – oder MAR über andere, nicht getestete Variablen.

**Gesamtbewertung der Missingness:**

Die systematischen Zusammenhänge mit Alter und Biopsy zeigen, dass die fehlenden Werte nicht vollständig zufällig (MCAR) sind. Ein einfaches listenweises Löschen würde zu Bias und Informationsverlust führen. Die Daten sind wahrscheinlich MAR (Missing At Random) oder MNAR (Missing Not At Random).


## Wichtigste Erkenntnisse
- Boxplots der Features mit Farbkodierung nach Biopsy zeigen, dass es keine klaren Entscheidungsgrenzen gibt.
- Biopsy hat kaum einen linearen Zusammenhang mit den Features.
- RandomForest mit RFECV (Recursive Feature Elimination with Cross-Validation) konnte die Anzahl der Features von 35 auf 19 reduzieren (45.7% Reduktion) bei gleichbleibend hoher Accuracy.
- Die Feature Selection durch RFECV verbessert die Generalisierung und reduziert Overfitting.
- RandomForest-Classifier mit RFECV ist signifikant besser für die Klassifizierung als ein KNN-Classifier (McNemar-Test, p < 0.05).
- Dimensionsreduktion mit PCA behält etwa 76% der Informationen.

### Erhöhtes Risiko für Cervical Cancer
- Diagnose Tests wie Schiller, Hinselmann und Cytology (stark)
- Dx: CIN (stark)
- Kürzlich diagnostizierte STDs (mittel)
- Lange Nutzung von hormonellen Verhütungsmitteln (mittel)
- Eine HPV Diagnose (mittel, aber Entscheidungsgrenze ausgeprägt)
- Je mehr Schwangerschaften die Person schon hatte (schwach)
- Je älter die Patientin ist (schwach)
- Je mehr STDs man schon hatte (schwach)
- Spiralen zur Verhütung (IUD) (schwach)
- Rauchen (Anzahl Schachteln pro Jahr und Jahre) (schwach, aber Entscheidungsgrenze ersichtlich)


## Technik
- Python 3.13.12
- Module in requirements.txt
- 'Project.ipynb' – vollständiger Analyse-Workflow
- main.py zum Ausführen von Streamlit
- ml.py für Logik

## Autor
Marzio Allemann