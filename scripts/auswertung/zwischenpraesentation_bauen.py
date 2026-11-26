import copy
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.util import Emu, Pt

SRC = "praesentationen/01_einstiegspraesentation.pptx"
OUT = "praesentationen/02_zwischenpraesentation.pptx"
p = Presentation(SRC)
S = list(p.slides)
vorlage = S[3]                       # Datenfolie: Karte, Kopf, Text, Fußzeile, Quellenzeile
el = {sh.name: sh._element for sh in vorlage.shapes}
LAYOUT = vorlage.slide_layout


def ersetze(shape, alt, neu):
    for para in shape.text_frame.paragraphs:
        for r in para.runs:
            if alt in r.text:
                r.text = r.text.replace(alt, neu)


# Titel- und Schlussfolie
for sh in S[0].shapes:
    if sh.has_text_frame:
        ersetze(sh, "Einstiegspräsentation", "Zwischenpräsentation")
        ersetze(sh, "24.11.2026", "15.12.2026")
for sh in S[6].shapes:
    if sh.has_text_frame:
        ersetze(sh, "Einstiegspräsentation", "Zwischenpräsentation")
S[0].notes_slide.notes_text_frame.text = (
    "Zwischenstand meines Projekts: Erkennung von Hauptschild und Zusatzzeichen mit einem Multi-Task-Netz. "
    "Kurz zur Erinnerung Frage und Ansatz, dann die Ergebnisse: das Modell, die Lernkurve als Antwort auf die "
    "Forschungsfrage, eine Analyse je Klasse und was ich gegen die Risiken getan habe.")


def neue_folie(titel, notiz):
    s = p.slides.add_slide(LAYOUT)
    s.shapes.title.text = titel
    for name in ("Fußzeilenplatzhalter 1", "Foliennummernplatzhalter 2"):
        s.shapes._spTree.append(copy.deepcopy(el[name]))
    s.notes_slide.notes_text_frame.text = notiz
    return s


def kopie(s, name, x, y, w, h, text=None, groesse=None, fett=None):
    e = copy.deepcopy(el[name])
    s.shapes._spTree.append(e)
    sh = s.shapes[-1]
    sh.left, sh.top, sh.width, sh.height = Emu(x), Emu(y), Emu(w), Emu(h)
    if text is not None:
        tf = sh.text_frame
        zeilen = text if isinstance(text, list) else [text]
        p0 = tf.paragraphs[0]
        for extra in tf.paragraphs[1:]:
            extra._p.getparent().remove(extra._p)
        r0 = p0.runs[0]
        for extra in p0.runs[1:]:
            extra._r.getparent().remove(extra._r)
        r0.text = zeilen[0]
        for z in zeilen[1:]:
            np_ = copy.deepcopy(p0._p)
            tf._txBody.append(np_)
            tf.paragraphs[-1].runs[0].text = z
        for para in tf.paragraphs:
            for r in para.runs:
                if groesse:
                    r.font.size = Pt(groesse)
                if fett is not None:
                    r.font.bold = fett
    return sh


def karte(s, x, y, w, h, kopf, text, groesse=14):
    kopie(s, "Karte Synset", x, y, w, h)
    kopie(s, "Kopf Synset", x + 228600, y + 152400, w - 457200, 369332, kopf)
    kopie(s, "Text Synset", x + 228600, y + 560000, w - 457200, h - 700000, text, groesse)


def zahl(s, x, y, w, wert, label):
    kopie(s, "Karte Synset", x, y, w, 1500000)
    kopie(s, "Kopf Synset", x + 150000, y + 150000, w - 300000, 700000, wert, 36)
    kopie(s, "Text Synset", x + 150000, y + 880000, w - 300000, 550000, label, 13)


def quelle(s, text):
    kopie(s, "Quellen Folie", 825500, 6070600, 10350500, 369332, text)


def stil(chart):
    chart.font.size = Pt(12)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False



# Standardabweichung über 3 Seeds (nur 25, 50, 150; übrige Stufen ein Lauf -> 0)
STD = [(0.025, 0.005, 0, 0.007, 0), (0.026, 0.008, 0, 0.010, 0)]


def fehlerbalken(ser, werte):
    """Fehlerbalken ± werte als natives c:errBars (python-pptx kann das nicht direkt)."""
    from lxml import etree
    C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    lit = "".join(f'<c:pt idx="{k}"><c:v>{v}</c:v></c:pt>' for k, v in enumerate(werte))
    num = f'<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="{len(werte)}"/>{lit}</c:numLit>'
    xml = (f'<c:errBars xmlns:c="{C}" xmlns:a="{A}"><c:errDir val="y"/><c:errBarType val="both"/>'
           f'<c:errValType val="cust"/><c:noEndCap val="0"/><c:plus>{num}</c:plus><c:minus>{num}</c:minus>'
           f'<c:spPr><a:ln w="19050"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill></a:ln></c:spPr></c:errBars>')
    e = etree.fromstring(xml)
    sx = ser._element
    nach = sx.find(f"{{{C}}}cat")          # Schema: errBars steht vor c:cat
    nach.addprevious(e)


L, B = 825500, 10541000   # linker Rand, Inhaltsbreite

# 1) Daten aufbereitet
s = neue_folie("Stand der Daten",
    "Seit der Einstiegspräsentation sind die echten Daten fertig. 651 Testbilder aus Aachen, der Eifel und Köln, "
    "die nie im Training vorkommen. Gut 4000 Trainingsausschnitte aus dem übrigen Deutschland, davon mehr als die "
    "Hälfte ohne Zusatzzeichen. Bei 558 Ausschnitten kannte Mapillary das Hauptschild nicht; die habe ich "
    "nachgelabelt, jetzt haben 620 der 651 Testbilder ein Hauptschild-Label statt 490.")
w = (B - 3 * 228600) // 4
for i, (wert, label) in enumerate([("651", "echte Testbilder (Aachen, Eifel, Köln)"),
                                   ("4.009", "echte Trainingsausschnitte"),
                                   ("2.199", "davon ohne Zusatzzeichen"),
                                   ("558", "Hauptschilder nachgelabelt")]):
    zahl(s, L + i * (w + 228600), 1450000, w, wert, label)
karte(s, L, 3200000, B, 2000000, "Was sich geändert hat",
      ["Test nach Ort getrennt: kein Foto aus Test-Regionen im Training",
       "Hauptschild-Label im Test: 620 statt 490 von 651 Bildern",
       "26 Zusatzzeichen-Klassen, Zielgruppe per StVO-Tabelle",
       "Synset liefert nur Hauptschild-Labels; Zusatzzeichen lernt das Netz nur aus echten Fotos"])
quelle(s, "Quellen: [2] Mapillary, Fotos CC BY-SA 4.0 · [6] A. Sielemann et al., IEEE ITSC 2024")

# 2) Ergebnis Modell A
s = neue_folie("Ergebnis Modell A auf echten Testfotos",
    "Das ist Modell A nach 15 Epochen, gemessen nur auf den echten Testfotos. Das Hauptschild erkennt es fast immer, "
    "ob ein Zusatzschild da ist ebenfalls. Bei 87 Prozent der Bilder stimmen alle Zusatzzeichen exakt, und bei "
    "95 Prozent der Schilder, bei denen das Zusatzzeichen die Zielgruppe ändert, leitet es die richtige Zielgruppe ab. "
    "Der Hauptschild-Wert stammt noch von 490 Testbildern; der Lauf mit allen 620 folgt.")
w = (B - 3 * 228600) // 4
for i, (wert, label) in enumerate([("98,8 %", "Hauptschild (490 Bilder*)"),
                                   ("99,5 %", "Zusatzschild vorhanden"),
                                   ("87,3 %", "alle Zusatzzeichen exakt"),
                                   ("94,6 %", "Zielgruppe korrekt¹")]):
    zahl(s, L + i * (w + 228600), 1450000, w, wert, label)
karte(s, L, 3200000, B, 2000000, "Einordnung",
      ["Micro-F1 Zusatzzeichen 0,92, Makro-F1 nur 0,56 (Grund: Folie 7)",
       "Gegenmaßnahme „keins“ auf 300 begrenzt: +0,6 bis +1,4 Punkte, im Rauschen → nicht übernommen",
       "* neuer Lauf mit 620 Hauptschild-Labels steht noch aus",
       "¹ nur Schilder, bei denen das Zusatzzeichen die Zielgruppe ändert"])
quelle(s, "Quelle: [3] K. He et al., IEEE CVPR 2016 (ResNet-18)")

# 3) Lernkurve
s = neue_folie("Antwort: ab etwa 50 echten Bildern",
    "Das ist die Antwort auf die Forschungsfrage. Ich trainiere dasselbe Modell mit höchstens 0, 25, 50, 100, 150 "
    "oder allen echten Bildern je Zusatzzeichen. Ohne echte Bilder geht gar nichts: Synset allein kennt die "
    "Zusatzzeichen nicht. Von 25 auf 50 steigt die Zielgruppe von 89 auf 94 Prozent, danach kommt kaum noch etwas "
    "dazu. 25, 50 und 150 habe ich mit drei Seeds wiederholt: ab 50 schwanken die Werte um weniger als einen Punkt. "
    "„Alle“ liegt etwas tiefer: Diese Stufen laufen nur 8 Epochen, und bei „alle“ ist über die Hälfte der echten "
    "Bilder ohne Zusatzzeichen. Jede Epoche zeigt dann weniger Zusatzzeichen-Beispiele. Mit 15 Epochen verschwindet "
    "der Effekt: Modell A mit allen Bildern und A mit höchstens 300 Bildern ohne Zusatz liegen gleichauf.")
cd = CategoryChartData()
cd.categories = ["25", "50", "100", "150", "alle"]
cd.add_series("Zielgruppe korrekt¹", (0.893, 0.937, 0.945, 0.951, 0.923))
cd.add_series("Zusatzzeichen exakt", (0.814, 0.859, 0.871, 0.871, 0.836))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Emu(L), Emu(1400000), Emu(7000000), Emu(4500000), cd)
ch = gf.chart
stil(ch)
ch.value_axis.maximum_scale, ch.value_axis.minimum_scale = 1.0, 0.75
ch.value_axis.major_unit = 0.05
ch.value_axis.tick_labels.number_format, ch.value_axis.tick_labels.number_format_is_linked = "0%", False
ch.value_axis.has_major_gridlines = True
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "echte Trainingsbilder je Zusatzzeichen (höchstens)"
for i, ser in enumerate(ch.plots[0].series):
    ser.smooth = False
    ser.format.line.color.theme_color = MSO_THEME_COLOR.ACCENT_1 if i == 0 else MSO_THEME_COLOR.TEXT_1
    ser.format.line.width = Pt(2.5)
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1 if i == 0 else MSO_THEME_COLOR.TEXT_1
    ser.data_labels.position = XL_LABEL_POSITION.ABOVE if i == 0 else XL_LABEL_POSITION.BELOW
    ser.data_labels.number_format, ser.data_labels.number_format_is_linked = "0%", False
    ser.data_labels.font.size = Pt(10)
    ser.data_labels.show_value = True
    fehlerbalken(ser, STD[i])
pl = ch.plots[0]
karte(s, L + 7200000, 1400000, B - 7200000, 4500000, "3 Seeds (25 / 50 / 150)",
      ["Fehlerbalken: ± 1 Std über 3 Seeds", "Streuung Zielgruppe:", "25: ± 2,5 Punkte", "50: ± 0,5 Punkte", "150: ± 0,7 Punkte",
       "0, 100, alle: ein Lauf", "0 echte Bilder: 0 % (Synset allein reicht nicht)", "Achse beginnt bei 75 %", "¹ Zielgruppe nur bei Schildern, deren Zusatz sie ändert"], 13)

# 4) Klassenanalyse
s = neue_folie("Warum Makro-F1 nur 0,56 ist",
    "Micro-F1 ist mit 0,92 gut, Makro-F1 mit 0,56 schwach. Die Aufschlüsselung je Klasse zeigt den Grund: Vier "
    "Klassen haben kein einziges echtes Trainingsbild, zwei weitere nur fünf bis sieben. Die kann das Netz nicht "
    "lernen, weil Synset keine Zusatzzeichen-Labels liefert. Ohne die vier Klassen ohne Trainingsbilder liegt Makro-F1 "
    "bei etwa 0,70. Die Pfeile sind trotz 20 bis 60 Bildern mittelmäßig, hier vermute ich Verwechslungen der Richtung.")
werte = [("Abschleppen", 0, 0.0), ("Bei Nässe", 0, 0.0), ("Zeitangabe", 0, 0.0), ("Text (Anlieger frei)", 0, 0.0),
         ("Straßenbahn", 5, 0.0), ("Umweltzone frei", 7, 0.0), ("andere", 30, 0.2), ("Pfeil links", 33, 0.44),
         ("Entfernung", 36, 0.59), ("Pfeile beidseitig", 20, 0.61), ("Pfeil rechts", 60, 0.64), ("Bus", 132, 0.78),
         ("Bus frei", 109, 0.87), ("Schwerbehinderte", 128, 0.95), ("LKW", 151, 0.96), ("Rad frei", 190, 0.98),
         ("PKW", 136, 0.99), ("LKW, Bus, Anhänger", 317, 1.0)]
cd = CategoryChartData()
cd.categories = [f"{n} ({t})" for n, t, _ in werte]
cd.add_series("F1 auf echten Testfotos", [f for *_, f in werte])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Emu(L), Emu(1350000), Emu(7000000), Emu(4650000), cd)
ch = gf.chart
ch.font.size = Pt(10)
ch.has_legend = False
ch.value_axis.maximum_scale, ch.value_axis.minimum_scale = 1.0, 0.0
ch.value_axis.has_major_gridlines = True
ch.category_axis.reverse_order = True
ser = ch.plots[0].series[0]
ch.plots[0].gap_width = 40
for i, (_, t, _) in enumerate(werte):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_2 if t < 10 else MSO_THEME_COLOR.ACCENT_1
karte(s, L + 7200000, 1350000, B - 7200000, 4650000, "Befund",
      ["In Klammern: echte Trainingsbilder", "0–7 Bilder: Balken fehlen, nicht lernbar",
       "Micro-F1 0,92 · Makro-F1 0,56", "ohne Klassen ohne Trainingsbilder: ≈ 0,70",
       "neue Kennzahl: Makro-F1 über trainierbare Klassen"], 13)

# 5) Risiken
s = neue_folie("Risiken und Gegenmaßnahmen",
    "Vier Risiken aus dem Projektplan und was daraus geworden ist. Die Metriken sind mit Tests abgesichert. Das "
    "fehlende Hauptschild-Label habe ich durch Nachlabeln behoben. Die Begrenzung der Klasse ohne Zusatzzeichen "
    "brachte nichts Messbares. Und die Lernkurve beruht jetzt auf drei Seeds statt einem.")
zeilen = [("Risiko", "Gegenmaßnahme", "Stand"),
          ("Fehler in einer Metrik", "Kennzahlen ausgelagert, 16 Unit-Tests", "erledigt"),
          ("Hauptschild fehlt (Mapillary „unbekannt“)", "558 Ausschnitte nachgelabelt", "erledigt, 620/651"),
          ("„keins“ dominiert (2.199 von 4.009)", "höchstens 300 „keins“ verglichen", "geprüft, kein Gewinn"),
          ("Lernkurve aus nur einem Lauf", "Stufen 25/50/150 mit 3 Seeds", "erledigt, ± < 1 Punkt ab 50"),
          ("Seltene Klassen ohne echte Bilder", "Kennzahl über trainierbare Klassen", "offen")]
tb = s.shapes.add_table(len(zeilen), 3, Emu(L), Emu(1450000), Emu(B), Emu(3600000)).table
tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Emu(4000000), Emu(4000000), Emu(B - 8000000)
for r, z in enumerate(zeilen):
    for c, t in enumerate(z):
        cell = tb.cell(r, c)
        cell.text = t
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(14)
                run.font.bold = r == 0

# 6) Nächste Schritte
s = neue_folie("Nächste Schritte bis zum Abschluss",
    "Bis zum Abschluss im Januar: Modell A mit allen Hauptschild-Labels neu trainieren, die Pfeil-Klassen mit einer "
    "Verwechslungsmatrix untersuchen, Architektur B mit U-Net als Vergleich auswerten und daraus den Demonstrator "
    "bauen, der für ein Foto die vollständige Regel ausgibt.")
w = (B - 2 * 228600) // 3
for i, (kopf, text) in enumerate([
        ("Modell", ["A neu mit 620 Hauptschild-Labels", "Verwechslungsmatrix Pfeile", "Architektur B (U-Net) vs. A"]),
        ("Auswertung", ["Makro-F1 über trainierbare Klassen", "Lernkurve mit Fehlerbalken", "E1: synthetisch vs. echt"]),
        ("Demonstrator", ["Foto rein, Regel und Zielgruppe raus", "Abschluss 19./26.01.2027"])]):
    karte(s, L + i * (w + 228600), 1450000, w, 2400000, kopf, text)

# Reihenfolge: Titel, Frage, Ansatz, 6 neue Folien, Quellen, Dank; Folien 2 (Problem) und 4 (Daten) raus
lst = p.slides._sldIdLst
ids = list(lst)
alt, neu = ids[:7], ids[7:]
for e in ids:
    lst.remove(e)
for e in [alt[0], alt[2], alt[4]] + neu + [alt[5], alt[6]]:
    lst.append(e)
for e in (alt[1], alt[3]):
    p.part.drop_rel(e.rId)
p.save(OUT)
print("ok", len(p.slides))
