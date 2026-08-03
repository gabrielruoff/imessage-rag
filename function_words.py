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
# no digits/underscores. BOTH apostrophes are accepted, straight and curly
# (what iOS types). For tokenize() that is a no-op -- it normalizes curly to
# straight before matching -- but it is load-bearing for the one caller that
# runs this regex over RAW text, persona_draft._strip_ungrounded_terms: with
# the straight-only form "don't" split into "don" + "t", and an ungrounded
# "don" was stripped out of a draft leaving the apostrophe stranded ("'t").
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


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

# First-person SINGULAR pronouns -- the psycholinguistic "I" measure, kept
# apart from the plural WE_WORDS above and the second-person set below so the
# lexicon can split them (used by the conflict analyzer and baseline). Includes
# the texting contractions tokenize() keeps whole ("i'm", "i've") and their
# apostrophe-dropped variants ("im", "imma").
PRONOUN_I = frozenset("""
    i me my mine myself
    i'm i've i'll i'd i'ma
    im imma
""".split())

# Second-person pronouns (singular or plural). Includes texting shorthands
# ("u", "ur"), apostrophe-dropped contractions ("youre"), and group address
# ("yall", "y'all").
PRONOUN_YOU = frozenset("""
    you your yours yourself yourselves
    you're you've you'll you'd youre
    u ur yall y'all
""".split())

# Absolutist vocabulary (Al-Mosaiwi & Johnstone 2018): dichotomous,
# all-or-nothing words whose elevated rate tracks distress and, for the
# conflict analyzer, escalation. The canonical 19, plus a few texting variants
# the tokenizer would otherwise miss ("def"/"defo" for definitely, "totes" for
# totally, dropped-g "everythin"/"nothin").
ABSOLUTIST = frozenset("""
    absolutely all always complete completely constant constantly definitely
    entire ever every everyone everything full must never nothing totally whole
    def defo totes everythin nothin
""".split())

# High-frequency conversational English beyond the function words above:
# generic verbs, generic nouns, fillers, and texting affirmations. Used by
# the Wrapped report to filter "top words" down to vocabulary that is
# actually characteristic of the corpus ("ok"/"yeah"/"going" tell a reader
# nothing). Deliberately conservative -- pet names, profanity, and oddball
# words stay interesting and are NOT listed here.
COMMON_WORDS = STOPWORDS | frozenset("""
    get gets got gotten getting go goes going gone went come comes coming
    came want wants wanted wanting know knows knew knowing known think
    thinks thought thinking need needs needed needing make makes made
    making take takes took taken taking see sees saw seen seeing look
    looks looked looking say says said saying tell tells told telling talk
    talks talked talking call calls called calling text texts texted
    texting let lets letting try tries tried trying use uses used using
    work works worked working give gives gave given giving find finds
    found finding feel feels felt feeling put puts putting keep keeps kept
    keeping start starts started starting stop stops stopped stopping
    leave leaves leaving stay stays stayed staying wait waits waited
    waiting watch watches watched watching eat eats ate eaten eating sleep
    sleeps slept sleeping buy buys bought buying pay pays paid paying run
    runs ran running drive drives drove driving walk walks walked walking
    help helps helped helping ask asks asked asking send sends sent
    sending bring brings brought bringing meet meets met meeting hear
    hears heard hearing remember remembers remembered forget forgets
    forgot forgotten guess guessed hope hopes hoped hoping wish wishes
    wished mean means meant seem seems seemed happen happens happened
    happening turn turns turned move moves moved live lives lived living
    show shows showed shown read reads reading write writes wrote written
    love loves loved like likes liked want check checks checked checking
    play plays played playing end ends ended ending change changes changed
    talk done doing did does
    time times day days night nights week weeks weekend weekends month
    months year years today tomorrow tonight yesterday morning afternoon
    evening minute minutes min mins hour hours second seconds moment
    moments while thing things stuff people person guy guys man men woman
    women kid kids friend friends home house place places way ways lot
    lots bit bits point points part parts side sides name names phone
    school word words idea ideas plan plans money food dinner lunch
    breakfast car game games life world door room rooms

    good great nice bad better best worse worst fine well okay ok kk
    right wrong sure new old big small little long short early late soon
    ready busy tired happy sad easy hard fun funny cool awesome cute crazy
    weird real true whole last next first same different able
    yeah yea yes yep yup yuppp no nah nope not hey hi hello bye oh ah um
    uh hmm hm mm mhm wow omg oof welp huh eh damn dang gosh geez jeez
    lol lols lmao lmaoo lmfao haha hahah hahaha hehe rofl
    btw idk idc imo imho tbh fyi rn asap af ffs stg ngl fr frfr ong bc cuz
    pls plz please thanks thank thanks thx ty tysm np yw sorry sry oops
    my bad man bro bruh dude dawg fam yall y'all
    that'll there's here's what's who's how's when's where's why's
    should've would've could've must've i'ma imma lemme gimme dunno
    kinda gonna gotta wanna sorta oughta
    one two three four five six seven eight nine ten
    back why where when how still only ever also sound sounds
    http https www com org net
""".split())

# Cheap deterministic prefilter for annotate.py's two EVENT_KINDS. kNN
# distillation structurally cannot produce events (a source='distilled' row
# always has events_json='[]'), so events are exactly the label the hybrid tier
# must spend real LLM calls on -- and LLM time is the scarce resource it
# rations. These substrings pick the chunks worth spending it on; the LLM still
# makes the call, so this is a recall-first net, not a classifier. Lowercase
# substrings (SQLite LIKE folds ASCII case for free, and annotate_events.py
# matches raw), deliberately free of LIKE wildcards (% and _) so the query
# needs no ESCAPE clause.
#
# Precision/recall, measured 2026-07-27 over a 350k-message / 25k-chunk corpus,
# scored against the 450 chunks that corpus had already LLM-annotated (48.4% of
# them carry >= 1 event -- the annotator is liberal, which is the base rate to
# beat). The unit that matters is the CHUNK, not the message: a chunk spans ~14
# messages, so a pattern firing on 1% of messages lights up ~10% of chunks.
#  - The set selects 19.0% of chunks corpus-wide (20.5% inside the top-20
#    dyads) at 83.3% precision and 32.1% recall -- 1.72x the base rate per LLM
#    call. Recall is capped by the annotator counting bare affection ("Love
#    you", "Nice", "Ok") as an event; no lexicon reaches those, and they are
#    not the moments worth buying.
#  - "thank" is kept broad on purpose. It alone selects 14.3% of chunks (1.3%
#    of messages) and is dominated by perfunctory "ok thanks" -- yet measured
#    alone it is the single most precise pattern here (86.0%), and narrowing it
#    to "thank you"/"thanks for"/"thank u"/... drops recall 32.1% -> 25.2% for
#    NO precision gain (83.3% -> 82.1%). Dropping it entirely lands the yield
#    at 7.7% (what the hybrid plan guessed) but at 14.7% recall. A terse thanks
#    is gratitude expressed, and gratitude is half of EVENT_KINDS.
#  - Kept despite noise: "so excited"/"finally got"/"guess what" are
#    anticipation or news-prefaces rather than news, and "engaged"/"promot"/
#    "hired" also match "engaged with your core", "promotional", "we hired a
#    plumber" -- but each costs < 0.5% of chunks and the true positives are
#    top-tier events with no other lexical signature.
#  - Dropped as hopeless: "passed" and every narrowing of it -- 265 messages,
#    of which the exam sense was ~0 ("passed out", "passed the exit", and
#    "passed away", which points at grief, the opposite of good news);
#    "accepted" (mostly automated tax/appointment notices); "i made it"
#    ("made it home safe"); "we got" and "i got the" (1.7% and 0.8% of chunks,
#    almost all "we got food"); "aced"/"ily"/"i won" (substrings of
#    placed/faced, family/easily, i won't).
#  - "re the best" is not a typo: it is the one wildcard-free substring that
#    catches "you're the best", "you’re the best" (the curly apostrophe iOS
#    actually types, and the reason the ASCII form matches almost nothing) and
#    "youre the best" at once.
# Pool size is not the run's bound -- Phase 3a's time cap is -- so the filter's
# job is to raise the event density of what the budget reaches, not to shrink
# the pool to the budget.
EVENT_CANDIDATE_PATTERNS = {
    "gratitude": (
        "thank", "thx", "thnx", "tysm", "ty sm", "appreciate", "grateful",
        "re the best", "owe you", "owe u", "means a lot",
    ),
    "good_news": (
        "congrat", "great news", "good news", "big news", "guess what",
        "got the job", "got the offer", "got an offer", "new job", "hired",
        "i got in", "we got in", "got accepted", "promot", "engaged",
        "pregnant", "graduated", "so excited", "so proud", "proud of you",
        "proud of u", "happy for you", "finally got", "we did it",
        # Praise responses: the other half of a good-news moment (and what
        # fills the prompt's response_class field) is the reply to it.
        "good job", "nice job", "well done",
    ),
}

# GoEmotions taxonomy (Demszky et al. 2020), minus 'neutral' -- the label
# set annotate.py's output is validated against.
EMOTION_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise",
)
