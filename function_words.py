"""Bundled open function-word lexicon and text-signal regexes, shared by the
lexicon pipeline (lexicon_build.py), the language analytics tools, and
annotate.py. Data-as-code: hand-compiled open word lists in the 9 LSM
(language style matching) function-word categories -- word lists are not the
licensed part of LIWC, and open lists reproduce LSM adequately (Ireland &
Pennebaker 2010 use exactly these category definitions).
"""

import re

FUNCTION_WORDS = {
    # Personal pronouns.
    "pronoun": frozenset("""
        i me my mine myself we us our ours ourselves you your yours yourself
        yourselves he him his himself she her hers herself they them their
        theirs themselves u ur
    """.split()),
    # Impersonal pronouns.
    "ipron": frozenset("""
        it its itself this that these those something anything nothing
        everything somebody anybody nobody everybody someone anyone
        everyone one another other others whatever whichever whoever
        stuff thing things what which who whom whose
    """.split()),
    "article": frozenset("a an the".split()),
    # Conjunctions (coordinating + common subordinating).
    "conj": frozenset("""
        and but or nor so yet for because although though while whereas
        unless until since if whether once whenever wherever plus also
        however altho cuz cause bc
    """.split()),
    "prep": frozenset("""
        about above across after against along amid among around at before
        behind below beneath beside besides between beyond by despite down
        during except from in inside into near of off on onto out outside
        over past through throughout till to toward towards under underneath
        up upon via with within without
    """.split()),
    # Auxiliary verbs (incl. common contractions post-tokenize).
    "aux": frozenset("""
        am is are was were be been being have has had having do does did
        doing will would shall should can could may might must ought
        i'm you're he's she's it's we're they're i've you've we've they've
        i'll you'll he'll she'll we'll they'll i'd you'd he'd she'd we'd
        they'd isn't aren't wasn't weren't hasn't haven't hadn't don't
        doesn't didn't won't wouldn't shan't shouldn't can't couldn't
        mightn't mustn't gonna gotta wanna
    """.split()),
    # Common adverbs (LIWC-style: high-frequency, largely non-content).
    "adverb": frozenset("""
        very really just quite rather too also again almost always never
        often sometimes usually rarely seldom soon still already yet even
        maybe perhaps probably definitely absolutely completely totally
        actually basically literally seriously honestly apparently
        especially exactly finally generally hardly mostly nearly now
        immediately eventually currently recently lately barely certainly
        clearly truly simply somewhat instead anyway anyways then there
        here well pretty super kinda sorta lowkey highkey
    """.split()),
    "negation": frozenset("""
        no not never none nothing nobody nowhere neither nor without
        don't doesn't didn't won't wouldn't can't couldn't shouldn't isn't
        aren't wasn't weren't hasn't haven't hadn't ain't cannot nah nope
    """.split()),
    # Quantifiers.
    "quant": frozenset("""
        all any both each every few fewer least less lot lots many more
        most much several some such enough plenty half double single
        entire whole bunch ton tons couple
    """.split()),
}

# Union of all 9 categories -- the "function word" stoplist used (together
# with a per-corpus top-200 list) to filter dyad-term candidates.
STOPWORDS = frozenset().union(*FUNCTION_WORDS.values())

# Word tokens: letter runs with an optional internal apostrophe ("don't"),
# no digits/underscores. Curly apostrophes (what iOS types) are normalized
# in tokenize() so contractions stay whole.
TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def tokenize(text):
    """Lowercased word tokens of `text` (see TOKEN_RE)."""
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


# Emoji, by the major Unicode emoji blocks (misc symbols+pictographs,
# emoticons, transport, supplemental+extended pictographs, plus the legacy
# dingbat/misc-symbol ranges).
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "☀-➿"
    "]"
)

# One-per-message laughter signal: standalone lol/lmao(o)/rofl/lmfao/ha(ha)/
# he(he) runs (with trailing letter stretch: lolll, hahahah), or a laughing
# emoji anywhere.
LAUGH_RE = re.compile(
    r"\b(?:l+o+l+o*|lmf?a+o+|rofl+|(?:ha){2,}h?|(?:he){2,}h?|ja+ja+j?a*)\w*"
    r"|[\U0001F602\U0001F923\U0001F606\U0001F604]",
    re.IGNORECASE,
)

WE_WORDS = frozenset("we us our ours ourselves we're we've we'll we'd lets let's".split())

# GoEmotions taxonomy (Demszky et al. 2020), minus 'neutral' -- the label
# set annotate.py's output is validated against.
EMOTION_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise",
)
