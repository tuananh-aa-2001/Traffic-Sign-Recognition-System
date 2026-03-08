"""
GTSRB Class Names and Descriptions
Maps class IDs (0-42) to traffic sign names in German and English
"""

GTSRB_CLASSES = {
    0: {'de': 'Geschwindigkeitsbegrenzung (20km/h)', 'en': 'Speed limit (20km/h)'},
    1: {'de': 'Geschwindigkeitsbegrenzung (30km/h)', 'en': 'Speed limit (30km/h)'},
    2: {'de': 'Geschwindigkeitsbegrenzung (50km/h)', 'en': 'Speed limit (50km/h)'},
    3: {'de': 'Geschwindigkeitsbegrenzung (60km/h)', 'en': 'Speed limit (60km/h)'},
    4: {'de': 'Geschwindigkeitsbegrenzung (70km/h)', 'en': 'Speed limit (70km/h)'},
    5: {'de': 'Geschwindigkeitsbegrenzung (80km/h)', 'en': 'Speed limit (80km/h)'},
    6: {'de': 'Ende der Geschwindigkeitsbegrenzung (80km/h)', 'en': 'End of speed limit (80km/h)'},
    7: {'de': 'Geschwindigkeitsbegrenzung (100km/h)', 'en': 'Speed limit (100km/h)'},
    8: {'de': 'Geschwindigkeitsbegrenzung (120km/h)', 'en': 'Speed limit (120km/h)'},
    9: {'de': 'Überholverbot', 'en': 'No passing'},
    10: {'de': 'Überholverbot für Fahrzeuge über 3.5t', 'en': 'No passing for vehicles over 3.5 metric tons'},
    11: {'de': 'Vorfahrt an der nächsten Kreuzung', 'en': 'Right-of-way at the next intersection'},
    12: {'de': 'Vorfahrtsstraße', 'en': 'Priority road'},
    13: {'de': 'Vorfahrt gewähren', 'en': 'Yield'},
    14: {'de': 'Halt', 'en': 'Stop'},
    15: {'de': 'Verbot für Fahrzeuge aller Art', 'en': 'No vehicles'},
    16: {'de': 'Verbot für Fahrzeuge über 3.5t', 'en': 'Vehicles over 3.5 metric tons prohibited'},
    17: {'de': 'Einfahrt verboten', 'en': 'No entry'},
    18: {'de': 'Gefahrenstelle', 'en': 'General caution'},
    19: {'de': 'Gefährliche Kurve nach links', 'en': 'Dangerous curve to the left'},
    20: {'de': 'Gefährliche Kurve nach rechts', 'en': 'Dangerous curve to the right'},
    21: {'de': 'Doppelkurve', 'en': 'Double curve'},
    22: {'de': 'Unebene Fahrbahn', 'en': 'Bumpy road'},
    23: {'de': 'Schleudergefahr', 'en': 'Slippery road'},
    24: {'de': 'Verengte Fahrbahn rechts', 'en': 'Road narrows on the right'},
    25: {'de': 'Baustelle', 'en': 'Road work'},
    26: {'de': 'Lichtzeichenanlage', 'en': 'Traffic signals'},
    27: {'de': 'Fußgänger', 'en': 'Pedestrians'},
    28: {'de': 'Kinder', 'en': 'Children crossing'},
    29: {'de': 'Fahrräder kreuzen', 'en': 'Bicycles crossing'},
    30: {'de': 'Schnee- oder Eisglätte', 'en': 'Beware of ice/snow'},
    31: {'de': 'Wildwechsel', 'en': 'Wild animals crossing'},
    32: {'de': 'Ende aller Streckenverbote', 'en': 'End of all speed and passing limits'},
    33: {'de': 'Rechts abbiegen', 'en': 'Turn right ahead'},
    34: {'de': 'Links abbiegen', 'en': 'Turn left ahead'},
    35: {'de': 'Geradeaus', 'en': 'Ahead only'},
    36: {'de': 'Geradeaus oder rechts', 'en': 'Go straight or right'},
    37: {'de': 'Geradeaus oder links', 'en': 'Go straight or left'},
    38: {'de': 'Rechts vorbei', 'en': 'Keep right'},
    39: {'de': 'Links vorbei', 'en': 'Keep left'},
    40: {'de': 'Kreisverkehr', 'en': 'Roundabout mandatory'},
    41: {'de': 'Ende des Überholverbots', 'en': 'End of no passing'},
    42: {'de': 'Ende des Überholverbots für Fahrzeuge über 3.5t', 'en': 'End of no passing by vehicles over 3.5 metric tons'}
}


def get_class_name(class_id: int, lang: str = 'en') -> str:
    """
    Get the class name for a given class ID
    
    Args:
        class_id: Class ID (0-42)
        lang: Language ('en' or 'de')
    
    Returns:
        Class name string
    """
    if class_id not in GTSRB_CLASSES:
        return f"Unknown class {class_id}"
    return GTSRB_CLASSES[class_id].get(lang, GTSRB_CLASSES[class_id]['en'])


def get_all_class_names(lang: str = 'en') -> list:
    """
    Get all class names in order
    
    Args:
        lang: Language ('en' or 'de')
    
    Returns:
        List of class names
    """
    return [GTSRB_CLASSES[i][lang] for i in range(43)]
