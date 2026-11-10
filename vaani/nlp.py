"""Rule-based incident understanding and TF-IDF duplicate detection.

Gnani transcribes speech. This module interprets that text locally. Suggestions
are investigative steps, not confirmed fixes.
"""

from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaani.languages import (
    follow_up_text,
    severity_reason_text,
    suggestion_disclaimer_text,
    troubleshooting_for_language,
    urgency_reason_text,
)

SUGGESTION_DISCLAIMER = (
    "These are investigation suggestions based on the reported symptoms. "
    "They are not a confirmed root cause or a guaranteed fix."
)

DUPLICATE_THRESHOLD = 0.38

URGENCIES = ("Immediate", "Urgent", "Today", "Scheduled")

CATEGORIES: list[tuple[str, list[tuple[str, int]]]] = [
    (
        "Security",
        [
            (r"ransomware|रैंसमवेयर|ర్యాన్సమ్‌వేర్|ரேன்சம்வேர்|ರ್ಯಾನ್ಸಮ್‌ವೇರ್|റാൻസംവെയർ|র‍্যানসমওয়্যার|રેન્સમવેર|ਰੈਨਸਮਵੇਅਰ", 5),
            (
                r"malware|phishing|data breach|\bbreach\b|virus|unauthorized access|"
                r"मैलवेयर|फ़िशिंग|डेटा चोरी|వైరస్|మాల్వేర్|ఫిషింగ్|வைரஸ்|மால்வேர்|ವೈರಸ್|മാൽവെയർ|ভাইরাস|વાયરસ|ਵਾਇਰਸ",
                4,
            ),
        ],
    ),
    (
        "Email & Collaboration",
        [
            (r"outlook|mailbox|exchange|e-?mail|microsoft teams|\bteams\b", 3),
            (
                r"ईमेल|इनबॉक्स|मेल|आउटलुक|மின்னஞ்சல்|இமெயில்|அவுட்லுக்|ఇమెయిల్|ఈమెయిల్|ఔట్‌లుక్|ಇಮೇಲ್|ಔಟ್‌ಲುಕ್|"
                r"ഇമെയിൽ|ഔട്ട്ലുക്ക്|ইমেল|ইমেইল|টিমস|ઇમેઇલ|આઉટલુક|ਈਮੇਲ|ਆਉਟਲੁੱਕ",
                3,
            ),
        ],
    ),
    (
        "Network & Connectivity",
        [
            (r"\bvpn\b|wi-?fi|wireless|ethernet|\bdns\b|internet|network|latency|packet", 3),
            (
                r"वीपीएन|नेटवर्क|इंटरनेट|वाईफाई|விபிஎன்|இணைக்க|இணைத்தல்|இணைப்பு|நெட்வொர்க்|இணையம்|வைஃபை|విపిఎన్|నెట్‌వర్క్|ఇంటర్నెట్|వైఫై|"
                r"ವಿಪಿಎನ್|ನೆಟ್‌ವರ್ಕ್|ಇಂಟರ್ನೆಟ್|വൈഫൈ|നെറ്റ്വർക്ക്|ওয়াইফাই|ઇન્ટરનેટ|વાઇફાઇ|ਵੀਪੀਐਨ|ਨੈੱਟਵਰਕ",
                3,
            ),
        ],
    ),
    (
        "Access & Identity",
        [
            (r"password|sign[\s-]?in|log[\s-]?in|sso|\bmfa\b|\botp\b|locked out|active directory|account locked", 3),
            (
                r"पासवर्ड|लॉगिन|கடவுச்சொல்|உள்நுழைவு|పాస్‌వర్డ్|లాగిన్|ಪಾಸ್‌ವರ್ಡ್|ಲಾಗಿನ್|പാസ്‌വേഡ്|ലോഗിൻ|"
                r"পাসওয়ার্ড|লগইন|પાસવર્ડ|લોગિન|ਪਾਸਵਰਡ|ਲੌਗਇਨ",
                3,
            ),
        ],
    ),
    (
        "Database",
        [
            (r"database|oracle|postgres|mysql|sql server|deadlock|\bquery\b|\bqueries\b", 3),
            (r"डेटाबेस|டேட்டாபேஸ்|தரவுத்தளம்|డేటాబేస్|డేటా బేస్|ಡೇಟಾಬೇಸ್|ഡാറ്റാബേസ്|ডেটাবেস|ડાટાબેઝ|ਡਾਟਾਬੇਸ", 3),
        ],
    ),
    (
        "Business Application",
        [
            (r"\bsap\b|salesforce|\bjira\b|portal|website|application crash|app crash|http\s*5\d\d|पोर्टल|వెబ్‌సైట్|வலைத்தளம்|போர்ட்டல்", 3),
        ],
    ),
    (
        "Hardware",
        [
            (r"printer|laptop|monitor|keyboard|mouse|\bdock\b|headset", 3),
            (
                r"प्रिंटर|लैपटॉप|பிரிண்டர்|லேப்டாப்|ప్రింటర్|ల్యాప్‌టాప్|ಪ್ರಿಂಟರ್|ಲ್ಯಾಪ್‌ಟಾಪ್|പ്രിന്റർ|ലാപ്‌ടോപ്പ്|"
                r"প্রিন্টার|ল্যাপটপ|પ્રિન્ટર|લેપટોપ|ਪ੍ਰਿੰਟਰ|ਲੈਪਟਾਪ",
                3,
            ),
        ],
    ),
    (
        "Server & Infrastructure",
        [
            (r"\bserver\b|outage|disk full|data cent(?:er|re)|reboot", 2),
            (r"सर्वर|சர்வர்|సర్వర్|ಸರ್ವರ್|സെർവർ|সার্ভার|સર્વર|ਸਰਵਰ", 2),
        ],
    ),
]

SERVICE_PATTERNS: list[tuple[str, str]] = [
    ("Outlook", r"outlook|mailbox|exchange|आउटलुक|அவுட்லுக்|ఔట్‌లుక్|ಔಟ್‌ಲುಕ್|ഔട്ട്ലുക്ക്|ਆਉਟਲੁੱਕ"),
    ("Microsoft Teams", r"microsoft teams|\bteams\b|टीम्स|டீம்ஸ்|టీమ్స్"),
    ("VPN", r"\bvpn\b|वीपीएन|விபிஎன்|విపిఎన్|ವಿಪಿಎನ್|വിപിഎൻ|ভিপিএন|વીપીએન|ਵੀਪੀਐਨ"),
    ("Email", r"\be-?mail\b|ईमेल|इनबॉक्स|मेल|மின்னஞ்சல்|இமெயில்|ఇమెయిల్|ఈమెయిల్|ಇಮೇಲ್|ഇമെയിൽ|ইমেল|ઇમેઇલ|ਈਮੇਲ"),
    ("Wi-Fi", r"wi-?fi|wireless|वाईफाई|வைஃபை|వైఫై|వై-ఫై|ವೈಫೈ|വൈഫൈ|ওয়াইফাই|વાઇફાઇ|ਵਾਈਫਾਈ"),
    ("Active Directory", r"active directory"),
    ("SAP", r"\bsap\b"),
    ("Database", r"database|oracle|postgres|mysql|sql server|डेटाबेस|டேட்டாபேஸ்|தரவுத்தளம்|డేటాబేస్|ಡೇಟಾಬೇಸ್|ഡാറ്റാബേസ്|ডেটাবেস|ડાટાબેઝ|ਡਾਟਾਬੇਸ"),
    ("Printer", r"printer|प्रिंटर|பிரிண்டர்|ప్రింటర్|ಪ್ರಿಂಟರ್|പ്രിന്റർ|প্রিন্টার|પ્રિન્ટર|ਪ੍ਰਿੰਟਰ"),
    ("Laptop", r"\blaptop\b|लैपटॉप|லேப்டாப்|ల్యాప్‌టాప్|ಲ್ಯಾಪ್‌ಟಾಪ್|ലാപ്‌ടോപ്പ്|ল্যাপটপ|લેપટોપ|ਲੈਪਟਾਪ"),
    ("File server", r"file server"),
    ("Web portal", r"\bportal\b|\bwebsite\b|पोर्टल|వెబ్‌సైట్|வலைத்தளம்"),
]

CRITICAL_PATTERNS: list[tuple[str, str]] = [
    (r"ransomware|रैंसमवेयर|ర్యాన్సమ్‌వేర్|ரேன்சம்வேர்|ರ್ಯಾನ್ಸಮ್‌ವೇರ್|റാൻസംവെയർ|র‍্যানসমওয়্যার|રેન્સમવેર|ਰੈਨਸਮਵੇਅਰ", "ransomware is mentioned"),
    (r"data loss|data breach|डेटा लॉस|डेटा चोरी|డేటా నష్టం|డేటా దొంగతనం|தரவு இழப்பு|தரவு திருட்டு|ಡೇಟಾ ನಷ್ಟ|ഡാറ്റ നഷ്ടം|ডেটা ক্ষতি|ડેટા નુકસાન|ਡਾਟਾ ਨੁਕਸਾਨ", "data loss or a breach is mentioned"),
    (r"malware outbreak", "a malware outbreak is mentioned"),
    (r"production\b.{0,60}\b(?:down|outage|unavailable)|प्रोडक्शन.{0,40}डाउन|ప్రొడక్షన్.{0,40}డౌన్|புரொடக்ஷன்.{0,40}டவுன்|ಪ್ರೊಡಕ್ಷನ್.{0,40}ಡೌನ್|പ്രൊഡക്ഷൻ.{0,40}ഡൗൺ|প্রোডাকশন.{0,40}ডাউন|પ્રોડક્શન.{0,40}ડાઉન|ਪ੍ਰੋਡਕਸ਼ਨ.{0,40}ਡਾਊਨ", "a production service is reported down"),
    (r"all users|entire company|whole company|company-wide|all customers|nobody in the company|सभी यूजर|पूरा सिस्टम|కంపెనీ అంతటా|వినియోగదారులందరికీ|నிறுவனம் முழுவதும்|அனைத்து பயனர்களும்|ಇಡೀ ಕಂಪನಿ|ಎಲ್ಲ ಬಳಕೆದಾರರು|കമ്പനി മുഴുവൻ|എല്ലാ ഉപയോക്താക്കൾക്കും|সমগ্র কোম্পানি|সমস্ত ব্যবহারকারী|આખી કંપની|બધા વપરાશકર્તાઓ|ਪੂਰੀ ਕੰਪਨੀ|ਸਾਰੇ ਵਰਤੋਂਕਾਰ", "the impact is described as organization-wide"),
    (r"complete outage|total outage|काम पूरी तरह ठप|పూర్తిగా డౌన్|ముழு முடக்கம்|ಸಂಪೂರ್ಣ ಸ್ಥಗಿತ|പൂർണ്ണ തടസ്സം|সম্পূর্ণ বন্ধ|સંપૂર્ણ ડાઉન|ਪੂਰੀ ਤਰ੍ਹਾਂ ਬੰਦ", "a complete outage is described"),
]

HIGH_PATTERNS: list[tuple[str, str]] = [
    (r"entire office|whole office|पूरा ऑफिस|మొత్తం ఆఫీస్|முழு அலுவலகம்|ಇಡೀ ಕಚೇರಿ|ഓഫീസ് മുഴുവൻ|পুরো অফিস|આખી ઑફિસ|ਪੂਰਾ ਦਫ਼ਤਰ", "the whole office is affected"),
    (r"whole team|several users|multiple users|many users|कई यूजर|పలువురు వినియోగదారులు|பல பயனர்கள்|ಅನೇಕ ಬಳಕೆದಾರರು|ഒന്നിലധികം ഉപയോക്താക്കൾ|একাধিক ব্যবহারকারী|ઘણા વપરાશકર્તાઓ|ਕਈ ਵਰਤੋਂਕਾਰ", "several people are affected"),
    (r"\bvpn\b.{0,20}\bdown\b|\bvpn is down\b|वीपीएन डाउन|విపిఎన్ డౌన్|விபிஎன் டவுன்|ವಿಪಿಎನ್ ಡೌನ್|വിപിഎൻ ਡൗൺ|ভিপিএন ডাউন|વીપીએન ડાઉન|ਵੀਪੀਐਨ ਡਾਊਨ", "VPN is reported down"),
    (r"\bnetwork\b.{0,20}\bdown\b|\bnetwork is down\b|नेटवर्क डाउन|నెట్‌వర్క్ డౌన్|நெட்வொர்க் டவுன்|ನೆಟ್‌ವರ್ಕ್ ಡೌನ್|നെറ്റ്വർക്ക് ഡൗൺ|নেটওয়ার্ক ডাউন|નેટવર્ક ડાઉન|ਨੈੱਟਵਰਕ ਡਾਊਨ", "the network is reported down"),
    (r"\bserver\b.{0,30}\b(?:down|unavailable|unreachable)|सर्वर डाउन|సర్వర్ డౌన్|சர்வர் டவுன்|ಸರ್ವರ್ ಡೌನ್|സെർവർ ഡൗൺ|সার্ভার ডাউন|સર્વર ડાઉન|ਸਰਵਰ ਡਾਊਨ", "a server is reported down"),
    (r"\bemail\b.{0,30}\bdown\b|ईमेल डाउन|ఇమెయిల్ డౌన్|இமெயில் டவுன்|ಇಮೇಲ್ ಡೌನ್|ഇമെയിൽ ഡൗൺ|ইমেল ডাউন|ઇમેઇલ ડાઉન|ਈਮੇਲ ਡਾਊਨ", "email is reported down"),
    (r"nobody can connect|cannot connect to the corporate network|काम नहीं कर रहा|పని చేయడం లేదు|வேலை செய்யவில்லை|யாரும்|ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ|പ്രവർത്തിക്കുന്നില്ല|কাজ করছে না|કામ કરતું નથી|ਕੰਮ ਨਹੀਂ ਕਰ ਰਿਹਾ", "people cannot connect or service is down"),
    (r"service is unavailable|service is down|severely degraded", "a service is unavailable"),
]

LOW_PATTERNS: list[tuple[str, str]] = [
    (r"printer is jammed|printer jammed|paper jam|प्रिंटर जाम|కాగితం ఇరుక్కుపోయింది|ప్రింటర్ జామ్|பிரிண்டர் ஜாம்|ಪ್ರಿಂಟರ್ ಜಾಮ್|പ്രിന്റർ ജാം|প্রিন্টার জ্যাম|પ્રિન્ટર જામ|ਪ੍ਰਿੰਟਰ ਜਾਮ", "this is a printer jam"),
    (r"password reset|पासवर्ड रीसेट|పాస్‌వర్డ్ రీసెట్|கடவுச்சொல் மீட்டமைக்க|ಪಾಸ್‌ವರ್ಡ್ ಮರುಹೊಂದಿಸಿ|പാസ്‌വേഡ് റീസെറ്റ്|পাসওয়ার্ড রিসেট|પાસવર્ડ રીસેટ|ਪਾਸਵਰਡ ਰੀਸੈੱਟ", "this is a password-reset request"),
    (r"how do i|how to\b|कैसे करें|ఎలా చేయాలి|எப்படி செய்வது|ಹೇಗೆ ಮಾಡುವುದು|ಎങ്ങനെ ചെയ്യാം|কীভাবে করব|કેવી રીતે કરવું|ਕਿਵੇਂ ਕਰੀਏ", "this is a how-to request"),
    (r"minor request|cosmetic", "the report describes a minor request"),
]

IMPACT_PATTERNS: list[tuple[str, str]] = [
    (r"entire company|whole company|all users|all customers|company-wide|nobody in the company|सभी यूजर|కంపెనీ అంతటా|వినియోగదారులందరికీ|నிறுவனம் முழுவதும்|அனைத்து பயனர்களும்|ಇಡೀ ಕಂಪನಿ|കമ്പനി മുഴുവൻ|সমগ্র কোম্পানি|આખી કંપની|ਪੂਰੀ ਕੰਪਨੀ", "Organization-wide"),
    (r"entire office|whole office|all employees|पूरा ऑफिस|మొత్తం ఆఫీస్|முழு அலுவலகம்|ಇಡೀ ಕಚೇರಿ|ഓഫീസ് മുഴുവൻ|পুরো অফিস|આખી ઑફિસ|ਪੂਰਾ ਦਫ਼ਤਰ", "Entire office"),
    (r"my team|several users|multiple users|many users|कई यूजर|పలువురు|பல பயனர்கள்|ಅನೇಕ ಬಳಕೆದಾರರು|ഒന്നിലധികം ആളുകൾ|একাধিক ব্যবহারকারী|ઘણા લોકો|ਕਈ ਵਰਤੋਂਕਾਰ", "Multiple users"),
    (r"only happening for me|only me|my own account|single user|\bfor me\b|केवल मेरे लिए|నా ఒక్కడికే|నాకు మాత్రమే|எனக்கு மட்டும்|ನನಗೆ ಮಾತ್ರ|എനിക്ക് മാത്രം|শুধু আমার|માત્ર મારા માટે|ਸਿਰਫ਼ ਮੇਰੇ ਲਈ", "Single user"),
]

TIME_PATTERN = r"\b(today|yesterday|this morning|since morning|just now|started|ago|from \d)\b|सुबह|कल|आज|இன்று|நேற்று|ఈరోజు|నిన్న|ఉదయం|ಇಂದು|ನಿನ್ನೆ|ഇന്ന്|ഇന്നലെ|আজ|গতকাল|આજે|ગઈકાલે|ਅੱਜ|ਕੱਲ੍ਹ"

ERROR_PATTERNS = [
    r"0x[0-9A-Fa-f]{4,8}",
    r"\bHTTP\s*[45]\d\d\b",
    r"\b[45]\d\d\b(?=\s+on\b|\s+error\b)",
    r"\b(?:ECONN\w+|ETIMEDOUT|ERR_[A-Z0-9_]+)\b",
    r"(?i)\b(?:error|exception|errcode|code)\s*[:#-]?\s*[A-Z0-9][A-Z0-9_.\-]{2,}",
]

ERROR_CATEGORIES = {
    "Email & Collaboration",
    "Business Application",
    "Server & Infrastructure",
    "Database",
    "Network & Connectivity",
}

PLAYBOOKS: dict[str, list[str]] = {
    "Email & Collaboration": [
        "Check whether the mail problem is one mailbox or the whole service.",
        "Confirm the device is online and can open the mail provider's status or login page.",
        "Write down the full error code before changing the mail profile.",
        "If mail is queued during an outage, avoid sending the same message many times.",
    ],
    "Network & Connectivity": [
        "Check whether ordinary websites open, or only the office network or VPN fails.",
        "Confirm the VPN or Wi-Fi profile is the one issued by the organization.",
        "Try one other network before treating the laptop as faulty.",
        "If many people are affected, report it as a shared service incident and avoid repeated password attempts.",
    ],
    "Access & Identity": [
        "Check whether the account locked after several sign-in attempts.",
        "Confirm the device clock is correct when a one-time code is rejected.",
        "Use the organization's reset process. Do not share the password in chat or email.",
        "Record the exact text shown after the failed sign-in.",
    ],
    "Database": [
        "Note which application and which query or screen fails.",
        "Check whether the database is unreachable or only one report is slow.",
        "Capture the error text, including any timeout or deadlock wording.",
        "Avoid rerunning a heavy query repeatedly while the database is already failing.",
    ],
    "Business Application": [
        "Record the application name, the screen, and the time of the failure.",
        "Check whether a refresh or another user account sees the same error.",
        "Save the HTTP status or error code shown on the page.",
        "If the portal is down for many people, wait for the service owner before clearing local data.",
    ],
    "Hardware": [
        "Note the device, the location, and what changed just before the fault.",
        "Check cables, power, and the selected printer or display.",
        "For a paper jam, follow the printer's own jam diagram rather than forcing the tray.",
        "If the device is under a service contract, raise that asset identifier with the desk.",
    ],
    "Security": [
        "Disconnect the affected machine from the network if ransomware or an active compromise is suspected.",
        "Do not pay a ransom or delete files that may be needed for investigation.",
        "Tell the security contact what was opened and which systems were reachable from that machine.",
        "Leave the machine powered as the security team instructs so evidence is preserved.",
    ],
    "Server & Infrastructure": [
        "Confirm which server or service name users cannot reach.",
        "Check whether the failure is a single host or a wider outage.",
        "Capture the time it started and any error shown to users.",
        "Avoid repeated restarts until the service owner has looked at the current state.",
    ],
    "General IT": [
        "Describe the application or device, what was clicked, and what happened instead.",
        "Note whether anyone else has the same problem.",
        "Copy any error text exactly.",
        "Record when it last worked.",
    ],
}


def _search(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def classify_category(text: str) -> tuple[str, list[str]]:
    scores: list[tuple[int, str, list[str]]] = []
    for name, rules in CATEGORIES:
        score = 0
        labels: list[str] = []
        for pattern, weight in rules:
            if _search(pattern, text):
                score += weight
                labels.append(pattern)
        if score:
            scores.append((score, name, labels))
    if not scores:
        return "General IT", []
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[0][1], scores[0][2]


def extract_service(text: str) -> str:
    for name, pattern in SERVICE_PATTERNS:
        if _search(pattern, text):
            return name
    return ""


def extract_errors(text: str) -> list[str]:
    found: list[str] = []
    for pattern in ERROR_PATTERNS:
        found.extend(match.group(0).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return _unique(found)[:5]


def extract_impact(text: str) -> str:
    for pattern, label in IMPACT_PATTERNS:
        if _search(pattern, text):
            return label
    return ""


def extract_technical_details(text: str, errors: list[str]) -> str:
    extras: list[str] = []
    extras.extend(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))
    extras.extend(re.findall(r"(?i)\bport\s+\d{2,5}\b", text))
    extras.extend(re.findall(r"(?i)\bwindows\s+\d+\b", text))
    parts = _unique([*errors, *extras])
    return "; ".join(parts)


def urgency_for_severity(severity: str, language_code: str = "en-IN") -> tuple[str, str]:
    table = {
        "Critical": "Immediate",
        "High": "Urgent",
        "Medium": "Today",
        "Low": "Scheduled",
    }
    urgency = table.get(severity, "Today")
    if language_code in ("en-IN", "en", ""):
        reasons = {
            "Immediate": "Respond now. Critical reports cover security events and organization-wide outages.",
            "Urgent": "Respond within the hour. A shared service or many people are affected.",
            "Today": "Handle today. The report is a real fault without a wide outage signal.",
            "Scheduled": "Queue this with routine requests. It reads as a minor or how-to issue.",
        }
        reason = reasons.get(urgency, "Handle today.")
    else:
        reason = urgency_reason_text(urgency, language_code)
    return urgency, reason


def classify_severity(text: str, language_code: str = "en-IN") -> tuple[str, str]:
    critical = [label for pattern, label in CRITICAL_PATTERNS if _search(pattern, text)]
    high = [label for pattern, label in HIGH_PATTERNS if _search(pattern, text)]
    low = [label for pattern, label in LOW_PATTERNS if _search(pattern, text)]
    if critical:
        sev = "Critical"
        if language_code in ("en-IN", "en", ""):
            reason = "Classified as Critical because " + "; ".join(_unique(critical)) + "."
            if high or low:
                reason += " Broader impact signals take priority over narrower ones."
        else:
            reason = severity_reason_text(sev, language_code)
        return sev, reason
    if high:
        sev = "High"
        if language_code in ("en-IN", "en", ""):
            reason = "Classified as High because " + "; ".join(_unique(high)) + "."
        else:
            reason = severity_reason_text(sev, language_code)
        return sev, reason
    if low:
        sev = "Low"
        if language_code in ("en-IN", "en", ""):
            reason = "Classified as Low because " + "; ".join(_unique(low)) + ". No organization-wide outage or security signal was found."
        else:
            reason = severity_reason_text(sev, language_code)
        return sev, reason
    sev = "Medium"
    if language_code in ("en-IN", "en", ""):
        reason = "Classified as Medium because the report does not describe an organization-wide outage, a security incident, or a minor how-to request."
    else:
        reason = severity_reason_text(sev, language_code)
    return sev, reason


def make_title(category: str, service: str, severity: str, text: str) -> str:
    if _search(r"password reset|पासवर्ड रीसेट|పాస్‌వర్డ్ రీసెట్|கடவுச்சொல் மீட்டமைக்க", text):
        return "Password reset request"
    if _search(r"printer is jammed|printer jammed|paper jam|प्रिंटर जाम|కాగితం ఇరుక్కుపోయింది|ప్రింటర్ జామ్|பிரிண்டர் ஜாம்", text):
        return "Printer jam"
    symptom = {"Critical": "outage", "High": "disruption", "Medium": "issue", "Low": "request"}[severity]
    subject = service or category
    return f"{subject} {symptom}"[:90]


def missing_questions(text: str, category: str, service: str, errors: list[str], impact: str, language_code: str) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []

    def add(question_id: str) -> None:
        if len(questions) >= 4:
            return
        questions.append(
            {
                "id": question_id,
                "text": follow_up_text(question_id, language_code),
                "text_en": follow_up_text(question_id, "en-IN"),
            }
        )

    if len(" ".join(text.split())) < 12:
        add("details")
    if not service:
        add("service")
    if not impact:
        add("impact")
    if category in ERROR_CATEGORIES and not errors:
        add("error")
    if not _search(TIME_PATTERN, text):
        add("when")
    return questions


def troubleshooting(category: str, language_code: str = "en-IN") -> list[str]:
    steps = troubleshooting_for_language(category, language_code)
    if not steps:
        steps = list(PLAYBOOKS.get(category, PLAYBOOKS["General IT"]))
    return steps


def analyze_incident(text: str, language_code: str = "en-IN") -> dict[str, Any]:
    cleaned = " ".join((text or "").split())
    speech_language = language_code if language_code not in {"", "indic-auto"} else "en-IN"
    category, _hits = classify_category(cleaned)
    service = extract_service(cleaned)
    errors = extract_errors(cleaned)
    impact = extract_impact(cleaned)
    severity, reason = classify_severity(cleaned, speech_language)
    urgency, urgency_reason = urgency_for_severity(severity if cleaned else "Medium", speech_language)
    title = make_title(category, service, severity, cleaned) if cleaned else "Untitled incident"
    return {
        "title": title,
        "description": cleaned,
        "category": category,
        "affected_service": service,
        "error_messages": errors,
        "user_impact": impact,
        "technical_details": extract_technical_details(cleaned, errors),
        "severity": severity,
        "severity_reason": reason if cleaned else "No description was provided.",
        "urgency": urgency,
        "urgency_reason": urgency_reason if cleaned else "No description was provided.",
        "follow_ups": missing_questions(cleaned, category, service, errors, impact, speech_language),
        "suggestions": troubleshooting(category, speech_language),
        "suggestion_disclaimer": suggestion_disclaimer_text(speech_language),
    }


def find_duplicates(text: str, incidents: list[dict[str, Any]], threshold: float = DUPLICATE_THRESHOLD) -> list[dict[str, Any]]:
    """Return similar incidents using TF-IDF cosine similarity over title, description, and category."""
    query = " ".join((text or "").split())
    if len(query) < 5 or not incidents:
        return []
    corpus = [query]
    usable: list[dict[str, Any]] = []
    for incident in incidents:
        if incident.get("status") == "Duplicate":
            continue
        document = " ".join(
            str(incident.get(field) or "")
            for field in ("title", "description", "transcript", "category", "affected_service")
        ).strip()
        if not document:
            continue
        usable.append(incident)
        corpus.append(document)
    if len(corpus) < 2:
        return []
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1).fit_transform(corpus)
    except ValueError:
        return []
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    matches: list[dict[str, Any]] = []
    for incident, score in zip(usable, scores):
        similarity = float(score)
        if similarity < threshold:
            continue
        matches.append(
            {
                "id": incident.get("id"),
                "reference": incident.get("reference"),
                "title": incident.get("title"),
                "category": incident.get("category"),
                "severity": incident.get("severity"),
                "status": incident.get("status"),
                "is_sample": bool(incident.get("is_sample")),
                "similarity": round(similarity, 3),
            }
        )
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches[:5]
