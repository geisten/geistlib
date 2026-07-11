/*
 * agent_home.h — the home_bridge tool family: voice-style commands against a
 * Home Assistant instance, the geist way — the MODEL only routes (command vs
 * status vs reply); device, action, and value are parsed DETERMINISTICALLY
 * here against a host-supplied device registry. See docs/agent.md.
 *
 * Two tools along the read/write boundary (= the safety boundary):
 *   command tool  {"request"}: turn on/off, dim, set temperature, open/close
 *   status  tool  {"request"}: read a device / sensor state
 * The forced path lifts the whole user sentence into "request"; this file
 * does the rest — the same philosophy as stock_movers, scaled to a registry.
 *
 * Registry file (GEIST_HOME_REGISTRY, default ./home-registry.txt), one
 * device per line:
 *     entity_id | domain | alias phrase, alias phrase, ...
 * An alias PHRASE matches when ALL its words occur (word-prefix, case-
 * insensitive) in the request; phrases are alternatives. Give every light a
 * generic "licht"/"light" phrase so "schalte das Licht ein" with two lights
 * yields the deliberate AMBIGUITY answer (a clarifying observation listing
 * the candidates) instead of a guess. More matched words = more specific =
 * wins outright.
 *
 * Write-whitelist: light, switch, climate, cover, media_player. Garage doors
 * and alarm panels are REFUSED with a fixed observation even if present in the
 * registry. LOCKS take a dedicated confirmation flow: locking runs directly
 * (the safe direction), unlocking is two turns — challenge, then a literal
 * confirm word for the same entity within HOME_CONFIRM_TTL_S; the slot is a
 * file (one appliance request = one process) and one-shot. The model never
 * decides the security question. Sensors are read-only.
 *
 * Pronouns ("mach ES wieder aus"): the command tool remembers the last
 * successfully addressed entity in ctx (last_entity) and falls back to it
 * when the request carries a pronoun and no device matches.
 *
 * Collectives ("alle Lichter aus"): a collective word plus a MULTI-device
 * alias match executes on every matched writable device — never on locks.
 * Bare "alles" matches no alias and stays the safe no-device answer.
 * Relative climate ("mach es wärmer"): no number but a direction word moves
 * the current setpoint by a fixed step.
 *
 * ponytail: verbs/aliases are ASCII-prefix matched; German umlaut verbs are
 * listed in both spellings (öffne/Öffne) because bytewise ci-compare cannot
 * fold UTF-8. Values are integers/decimals lifted as the first number word.
 */
#ifndef GEIST_AGENT_HOME_H
#define GEIST_AGENT_HOME_H

#include <geist.h>

#include "agent.h"
#include "ha_rest.h"

#include <stdio.h>
#include <time.h> /* time() — the unlock-confirmation TTL */
#include <stdlib.h>
#include <string.h>
#include <strings.h>

enum { HOME_MAX_DEVICES = 64, HOME_OBS_CAP = 2048 };

struct home_device {
    char entity[64];
    char domain[16];
    char aliases[256]; /* comma-separated phrases */
};

struct home_ctx {
    const char        *ha_url;   /* nullptr -> ha_env_url() */
    const char        *ha_token; /* nullptr -> ha_env_token(); "" = unconfigured */
    struct home_device dev[HOME_MAX_DEVICES];
    size_t             n_dev;
    char               last_entity[64]; /* pronoun target ("mach es aus") */
    /* Unlock confirmation slot: a FILE, not memory — the appliance runs one
     * process per request, and the confirmation arrives in the next one.
     * nullptr -> GEIST_HOME_PENDING or ./.geist-home-pending. */
    const char *pending_path;
    /* transport hooks — the eval stub swaps these for a mutating state table */
    long (*set)(struct home_ctx          *c,
                const struct home_device *d,
                const char               *service,
                const char               *extra,
                size_t                    cap,
                char                      out[]);
    long (*get)(struct home_ctx *c, const struct home_device *d, size_t cap, char out[]);
};

/* ---- registry -------------------------------------------------------------- */

/* "entity | domain | aliases" -> dev. Returns 1 on a usable line. PURE. */
static inline int home_registry_line(const char *line, struct home_device *d) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
        return 0;
    }
    const char *p1 = strchr(line, '|');
    const char *p2 = p1 ? strchr(p1 + 1, '|') : nullptr;
    if (p1 == nullptr || p2 == nullptr) {
        return 0;
    }
#define HOME_TRIM_INTO(dst, s, e)                                                \
    do {                                                                         \
        const char *s_ = (s);                                                    \
        const char *e_ = (e);                                                    \
        while (s_ < e_ && (*s_ == ' ' || *s_ == '\t')) {                         \
            s_++;                                                                \
        }                                                                        \
        while (e_ > s_ && (e_[-1] == ' ' || e_[-1] == '\t' || e_[-1] == '\n')) { \
            e_--;                                                                \
        }                                                                        \
        size_t n_ = (size_t) (e_ - s_);                                          \
        if (n_ >= sizeof(dst)) {                                                 \
            n_ = sizeof(dst) - 1;                                                \
        }                                                                        \
        memcpy(dst, s_, n_);                                                     \
        (dst)[n_] = '\0';                                                        \
    } while (0)
    HOME_TRIM_INTO(d->entity, line, p1);
    HOME_TRIM_INTO(d->domain, p1 + 1, p2);
    HOME_TRIM_INTO(d->aliases, p2 + 1, p2 + 1 + strlen(p2 + 1));
#undef HOME_TRIM_INTO
    return d->entity[0] != '\0' && d->domain[0] != '\0' && d->aliases[0] != '\0';
}

/* Load GEIST_HOME_REGISTRY (or `path`) into ctx. Returns device count. */
static inline size_t home_registry_load(struct home_ctx *c, const char *path) {
    if (path == nullptr || path[0] == '\0') {
        path = getenv("GEIST_HOME_REGISTRY");
    }
    if (path == nullptr || path[0] == '\0') {
        path = "./home-registry.txt";
    }
    c->n_dev = 0;
    FILE *f  = fopen(path, "r");
    if (f == nullptr) {
        return 0;
    }
    char line[512];
    while (c->n_dev < HOME_MAX_DEVICES && fgets(line, sizeof line, f)) {
        if (home_registry_line(line, &c->dev[c->n_dev])) {
            c->n_dev++;
        }
    }
    fclose(f);
    return c->n_dev;
}

/* ---- request parsing (pure) ------------------------------------------------ */

/* True if `word` occurs word-prefix, case-insensitively, in req. */
static inline int home_word_in(const char *req, const char *word) {
    size_t wl = strlen(word);
    for (const char *p = req; *p; p++) {
        if ((p == req || p[-1] == ' ' || p[-1] == '\t' || p[-1] == ',' || p[-1] == '.') &&
            strncasecmp(p, word, wl) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Match score of one alias phrase: all words present -> word count, else 0. */
static inline int home_phrase_score(const char *req, const char *phrase, size_t plen) {
    char buf[128];
    if (plen >= sizeof buf) {
        return 0;
    }
    memcpy(buf, phrase, plen);
    buf[plen] = '\0';
    int   n   = 0;
    char *save;
    for (char *w = strtok_r(buf, " \t", &save); w; w = strtok_r(nullptr, " \t", &save)) {
        if (!home_word_in(req, w)) {
            return 0;
        }
        n++;
    }
    return n;
}

/* Best alias-phrase score of a device against the request (0 = no match). */
static inline int home_device_score(const char *req, const struct home_device *d) {
    int         best = 0;
    const char *p    = d->aliases;
    while (*p) {
        const char *e = strchr(p, ',');
        size_t      n = e ? (size_t) (e - p) : strlen(p);
        int         s = home_phrase_score(req, p, n);
        if (s > best) {
            best = s;
        }
        p = e ? e + 1 : p + n;
        while (*p == ' ') {
            p++;
        }
    }
    return best;
}

/* All devices with the TOP score into idx[] (ties = deliberate ambiguity).
 * Returns the count; 0 = nothing matched. */
static inline size_t
home_match(const struct home_ctx *c, const char *req, size_t max, size_t idx[static max]) {
    int    best = 0;
    size_t n    = 0;
    for (size_t i = 0; i < c->n_dev; i++) {
        int s = home_device_score(req, &c->dev[i]);
        if (s > best) {
            best = s;
            n    = 0;
        }
        if (s == best && s > 0 && n < max) {
            idx[n++] = i;
        }
    }
    return best > 0 ? n : 0;
}

/* True if the request leans on a pronoun (es/it/ihn/sie at a word start). */
static inline int home_has_pronoun(const char *req) {
    return home_word_in(req, "es") || home_word_in(req, "it") || home_word_in(req, "sie") ||
           home_word_in(req, "ihn") || home_word_in(req, "them");
}

/* First number in the request ("21", "21.5") -> out. Returns 1 if found. */
static inline int home_parse_number(const char *req, size_t cap, char out[static cap]) {
    for (const char *p = req; *p; p++) {
        if ((*p >= '0' && *p <= '9') && (p == req || !(p[-1] >= '0' && p[-1] <= '9'))) {
            size_t w = 0;
            while ((*p >= '0' && *p <= '9') || (*p == '.' && p[1] >= '0' && p[1] <= '9')) {
                if (w + 1 < cap) {
                    out[w++] = *p;
                }
                p++;
            }
            out[w] = '\0';
            return w > 0;
        }
    }
    return 0;
}

/* Word lists: both German spellings where umlauts capitalize (see header). */
static inline int home_wants_off(const char *req) {
    static const char *const w[] = {"aus", "off", "ausschalten", "stopp", "stop"};
    for (size_t i = 0; i < sizeof w / sizeof *w; i++) {
        if (home_word_in(req, w[i])) {
            return 1;
        }
    }
    return 0;
}
static inline int home_wants_open(const char *req) {
    static const char *const w[] = {"öffne", "Öffne", "auf", "hoch", "open", "raise"};
    for (size_t i = 0; i < sizeof w / sizeof *w; i++) {
        if (home_word_in(req, w[i])) {
            return 1;
        }
    }
    return 0;
}
static inline int home_wants_close(const char *req) {
    static const char *const w[] = {"schließe", "schliesse", "zu", "runter", "close", "lower"};
    for (size_t i = 0; i < sizeof w / sizeof *w; i++) {
        if (home_word_in(req, w[i])) {
            return 1;
        }
    }
    return 0;
}

/* Relative climate direction: +1 warmer, -1 cooler, 0 neither. Word-initial
 * umlauts don't occur here, so one spelling per word plus the ae/ue forms. */
static inline int home_relative_dir(const char *req) {
    static const char *const warm[] = {"wärmer", "waermer", "warmer"};
    static const char *const cool[] = {
            "kälter", "kaelter", "kühler", "kuehler", "cooler", "colder"};
    for (size_t i = 0; i < sizeof warm / sizeof *warm; i++) {
        if (home_word_in(req, warm[i])) {
            return 1;
        }
    }
    for (size_t i = 0; i < sizeof cool / sizeof *cool; i++) {
        if (home_word_in(req, cool[i])) {
            return -1;
        }
    }
    return 0;
}

/* True if the request addresses a device GROUP ("alle Lichter", "all
 * lights"). "all" word-prefix also covers alle/alles. */
static inline int home_is_collective(const char *req) {
    return home_word_in(req, "all") || home_word_in(req, "sämtliche") ||
           home_word_in(req, "saemtliche") || home_word_in(req, "everything");
}

/* Map request + domain to an HA service + optional extra JSON field. Returns
 * service name or nullptr when the domain has no writable mapping. PURE. */
static inline const char *
home_action(const char *req, const char *domain, size_t cap, char extra[static cap]) {
    extra[0] = '\0';
    if (strcmp(domain, "light") == 0 || strcmp(domain, "switch") == 0 ||
        strcmp(domain, "media_player") == 0) {
        char num[16];
        if (strcmp(domain, "light") == 0 &&
            (home_word_in(req, "dimme") || home_word_in(req, "dim")) &&
            home_parse_number(req, sizeof num, num)) {
            snprintf(extra, cap, "\"brightness_pct\":%s", num);
            return "turn_on";
        }
        return home_wants_off(req) ? "turn_off" : "turn_on";
    }
    if (strcmp(domain, "climate") == 0) {
        char num[16];
        if (home_parse_number(req, sizeof num, num)) {
            snprintf(extra, cap, "\"temperature\":%s", num);
            return "set_temperature";
        }
        return nullptr; /* no target value -> not a command we can run */
    }
    if (strcmp(domain, "cover") == 0) {
        if (home_wants_close(req) && !home_wants_open(req)) {
            return "close_cover";
        }
        return "open_cover";
    }
    return nullptr; /* sensors etc. are read-only; locks & co. never mapped */
}

/* ---- real transports (HA REST) -------------------------------------------- */

static inline long home_set_ha(struct home_ctx          *c,
                               const struct home_device *d,
                               const char               *service,
                               const char               *extra,
                               size_t                    cap,
                               char                      out[]) {
    const char *url = c->ha_url ? c->ha_url : ha_env_url();
    const char *tok = c->ha_token ? c->ha_token : ha_env_token();
    if (tok[0] == '\0') {
        return -2; /* unconfigured */
    }
    return ha_call_service(url, tok, d->domain, service, d->entity, extra, cap, out);
}

static inline long
home_get_ha(struct home_ctx *c, const struct home_device *d, size_t cap, char out[]) {
    const char *url = c->ha_url ? c->ha_url : ha_env_url();
    const char *tok = c->ha_token ? c->ha_token : ha_env_token();
    if (tok[0] == '\0') {
        return -2;
    }
    return ha_get_state(url, tok, d->entity, cap, out);
}

/* ---- response language ----------------------------------------------------
 * The answer's device STATE words are TABLE-driven: one row per language, in a
 * fixed column order keyed by HA's raw state string. A new language is ONE ROW
 * here — no new branches anywhere. GEIST_HOME_LANG picks the row by a prefix
 * match on its code ("en", "en_US" both hit "en"); unset or unknown -> row 0
 * (German, the default the eval gate runs on).
 *
 * Cost: one getenv + a linear scan of a handful of rows/columns per answer —
 * nanoseconds against the ~2 s model turn, so no measurable hit. (No static
 * cache on purpose: the unit test flips the env mid-run.)
 *
 * Scope: the STATE words only — what every command/status answer shows. The
 * rarer clarify/challenge/error SENTENCES are still German literals below.
 * ponytail: table now covers the words a new language needs 90 % of; lift the
 * sentences into the same struct (a fmt[] column) when an English-FIRST
 * deployment actually ships, not before. */
enum {
    HOME_ST_ON,
    HOME_ST_OFF,
    HOME_ST_OPEN,
    HOME_ST_CLOSED,
    HOME_ST_LOCKED,
    HOME_ST_UNLOCKED,
    HOME_ST_UNAVAIL,
    HOME_ST_UNKNOWN,
    HOME_N_STATES
};
/* HA's raw state string for each column (the lookup key). */
static const char *const HOME_STATE_KEYS[HOME_N_STATES] = {
        "on", "off", "open", "closed", "locked", "unlocked", "unavailable", "unknown"};

struct home_lang {
    const char *code;
    const char *word[HOME_N_STATES];
};
/* One row per language. To add a language: copy a row, translate the words. */
static const struct home_lang HOME_LANGS[] = {
        {"de",
         {"an",
          "aus",
          "offen",
          "geschlossen",
          "abgeschlossen",
          "entriegelt",
          "nicht erreichbar",
          "unbekannt"}},
        {"en", {"on", "off", "open", "closed", "locked", "unlocked", "unavailable", "unknown"}},
};
enum { HOME_N_LANGS = sizeof HOME_LANGS / sizeof *HOME_LANGS };

/* Active language row from GEIST_HOME_LANG (prefix match); row 0 = default. */
static inline const struct home_lang *home_lang(void) {
    const char *l = getenv("GEIST_HOME_LANG");
    if (l && l[0]) {
        for (size_t i = 0; i < HOME_N_LANGS; i++) {
            if (strncasecmp(l, HOME_LANGS[i].code, strlen(HOME_LANGS[i].code)) == 0) {
                return &HOME_LANGS[i];
            }
        }
    }
    return &HOME_LANGS[0];
}

/* State word for an answer in the active language; unknown states pass through
 * verbatim (e.g. a numeric climate setpoint). */
static inline const char *home_state_word(const char *s) {
    const struct home_lang *L = home_lang();
    for (size_t i = 0; i < HOME_N_STATES; i++) {
        if (strcmp(s, HOME_STATE_KEYS[i]) == 0) {
            return L->word[i];
        }
    }
    return s;
}

/* Resolve the request to exactly one device, or write the clarify/error
 * observation and return -1. Pronouns fall back to last_entity. */
static inline long home_resolve(struct home_ctx *c,
                                const char      *req,
                                size_t           out_cap,
                                char             out[],
                                size_t          *out_len,
                                size_t          *dev_idx) {
    size_t idx[8];
    size_t n = home_match(c, req, 8, idx);
    if (n == 1) {
        *dev_idx = idx[0];
        return 0;
    }
    if (n == 0 && c->last_entity[0] != '\0' && home_has_pronoun(req)) {
        for (size_t i = 0; i < c->n_dev; i++) {
            if (strcmp(c->dev[i].entity, c->last_entity) == 0) {
                *dev_idx = i;
                return 0;
            }
        }
    }
    if (n == 0) {
        (void) agent_obs(out_cap,
                         out,
                         out_len,
                         "Kein Gerät passt zu dieser Anfrage. Bekannte Geräte stehen im "
                         "Registry (%zu Einträge).",
                         c->n_dev);
        return -1;
    }
    size_t w = (size_t) snprintf(out, out_cap, "Mehrere Geräte passen — welches ist gemeint?");
    for (size_t i = 0; i < n && w + 4 < out_cap; i++) {
        w += (size_t) snprintf(out + w,
                               out_cap - w,
                               " %s (%s)%s",
                               c->dev[idx[i]].entity,
                               c->dev[idx[i]].domain,
                               i + 1 < n ? "," : "");
    }
    if (out_len) {
        *out_len = w;
    }
    return -1;
}

/* ---- lock confirmation flow -------------------------------------------------
 * Locking runs directly (the SAFE direction — like turning a light off), but
 * UNLOCKING is two turns: the first request arms a pending slot and answers
 * with a challenge; only a follow-up that carries the confirm word and
 * resolves to the SAME entity executes. The check is entirely deterministic —
 * the model never decides a security question, it only ferries the user's
 * words. The slot is a file (one appliance request = one process), one-shot
 * (any other command disarms it), and expires after HOME_CONFIRM_TTL_S.
 * Garage doors and alarm panels stay refused outright — no flow. */
enum { HOME_CONFIRM_TTL_S = 120 };

/* "lock" (abschließen — direct), "unlock" (entriegeln — needs confirmation),
 * or nullptr when the request names neither direction. PURE. */
static inline const char *home_lock_action(const char *req) {
    /* stems end BEFORE the el/le metathesis: "entrieg" covers entriegeln,
     * entriegelt AND the imperative "Entriegle" */
    static const char *const unlock_w[] = {
            "entrieg", "aufschließ", "aufsperr", "unlock", "aufmach", "öffn", "open"};
    static const char *const lock_w[] = {
            "verrieg", "abschließ", "absperr", "zusperr", "zuschließ", "lock", "abmach"};
    for (size_t i = 0; i < sizeof unlock_w / sizeof *unlock_w; i++) {
        if (home_word_in(req, unlock_w[i])) {
            return "unlock";
        }
    }
    /* separable verbs: "schließ … auf" = unlock, "schließ … ab/zu" = lock */
    if (home_word_in(req, "schließ") || home_word_in(req, "sperr")) {
        if (home_word_in(req, "auf")) {
            return "unlock";
        }
        return "lock";
    }
    for (size_t i = 0; i < sizeof lock_w / sizeof *lock_w; i++) {
        if (home_word_in(req, lock_w[i])) {
            return "lock";
        }
    }
    return nullptr;
}

/* True if the request carries the literal confirm word. PURE. */
static inline int home_is_confirm(const char *req) {
    return home_word_in(req, "bestätige") || home_word_in(req, "bestaetige") ||
           home_word_in(req, "confirm");
}

static inline const char *home_pending_path(const struct home_ctx *c) {
    if (c->pending_path != nullptr) {
        return c->pending_path;
    }
    const char *p = getenv("GEIST_HOME_PENDING");
    return (p && p[0]) ? p : "./.geist-home-pending";
}

static inline void home_pending_clear(const struct home_ctx *c) {
    remove(home_pending_path(c));
}

/* Arm the slot: "<entity>|<service>|<epoch>". */
static inline int
home_pending_arm(const struct home_ctx *c, const char *entity, const char *service, long now) {
    FILE *f = fopen(home_pending_path(c), "w");
    if (f == nullptr) {
        return -1;
    }
    fprintf(f, "%s|%s|%ld\n", entity, service, now);
    fclose(f);
    return 0;
}

/* Take the slot if it matches entity+service and is fresh at `now` (injected
 * for testability). The slot is CONSUMED either way once it exists — a stale
 * or mismatched confirmation must not leave a live slot behind. Returns 1 on
 * a valid match, 0 otherwise. */
static inline int
home_pending_take(const struct home_ctx *c, const char *entity, const char *service, long now) {
    char  line[256] = "";
    FILE *f         = fopen(home_pending_path(c), "r");
    if (f == nullptr) {
        return 0;
    }
    const char *got = fgets(line, sizeof line, f);
    fclose(f);
    home_pending_clear(c);
    if (got == nullptr) {
        return 0;
    }
    char *p1 = strchr(line, '|');
    char *p2 = p1 ? strchr(p1 + 1, '|') : nullptr;
    if (p1 == nullptr || p2 == nullptr) {
        return 0;
    }
    *p1      = '\0';
    *p2      = '\0';
    long age = now - strtol(p2 + 1, nullptr, 10);
    return strcmp(line, entity) == 0 && strcmp(p1 + 1, service) == 0 && age >= 0 &&
           age <= HOME_CONFIRM_TTL_S;
}

/* ---- the two tools ---------------------------------------------------------- */

/* Write-whitelist — garage/alarm are refused BY OMISSION here and by the
 * explicit check below, even when the registry lists them. Locks are NOT in
 * this list either: they take the dedicated confirmation flow above. */
static inline int home_domain_writable(const char *domain) {
    return strcmp(domain, "light") == 0 || strcmp(domain, "switch") == 0 ||
           strcmp(domain, "climate") == 0 || strcmp(domain, "cover") == 0 ||
           strcmp(domain, "media_player") == 0;
}

/* The lock branch of the command tool: direct lock, challenge/confirm unlock.
 * Writes the observation; the caller has already resolved the device. */
static inline enum geist_status home_lock_flow(struct home_ctx          *c,
                                               const char               *req,
                                               const struct home_device *d,
                                               long                      now,
                                               size_t                    out_cap,
                                               char                      out[],
                                               size_t                   *out_len) {
    const char *action = home_lock_action(req);
    if (home_is_confirm(req)) {
        if (action == nullptr || strcmp(action, "unlock") != 0 ||
            !home_pending_take(c, d->entity, "unlock", now)) {
            return agent_obs(out_cap,
                             out,
                             out_len,
                             "Nichts zu bestätigen für %s (Anfrage abgelaufen oder nicht "
                             "gestellt).",
                             d->entity);
        }
        static char raw[HOME_OBS_CAP];
        long        rc = c->set(c, d, "unlock", "", sizeof raw, raw);
        if (rc < 0) {
            return agent_obs(
                    out_cap, out, out_len, "error: Home Assistant nicht erreichbar (unlock).");
        }
        snprintf(c->last_entity, sizeof c->last_entity, "%s", d->entity);
        return agent_obs(
                out_cap, out, out_len, "OK: %s → %s", d->entity, home_state_word("unlocked"));
    }
    if (action == nullptr) {
        return agent_obs(out_cap,
                         out,
                         out_len,
                         "Unklar, was mit %s geschehen soll (abschließen/aufschließen).",
                         d->entity);
    }
    if (strcmp(action, "lock") == 0) { /* the safe direction runs directly */
        static char raw[HOME_OBS_CAP];
        long        rc = c->set(c, d, "lock", "", sizeof raw, raw);
        if (rc < 0) {
            return agent_obs(
                    out_cap, out, out_len, "error: Home Assistant nicht erreichbar (lock).");
        }
        snprintf(c->last_entity, sizeof c->last_entity, "%s", d->entity);
        return agent_obs(
                out_cap, out, out_len, "OK: %s → %s", d->entity, home_state_word("locked"));
    }
    /* unlock without confirmation: arm the slot, answer with the challenge */
    home_pending_arm(c, d->entity, "unlock", now);
    return agent_obs(out_cap,
                     out,
                     out_len,
                     "Sicherheitsabfrage: %s wirklich entriegeln? Zum Bestätigen sage: "
                     "\"Bestätige entriegeln\" und das Gerät (gültig %d Sekunden).",
                     d->entity,
                     (int) HOME_CONFIRM_TTL_S);
}

static inline enum geist_status home_command_invoke(void      *ctx,
                                                    size_t     args_len,
                                                    const char args[static args_len],
                                                    size_t     out_cap,
                                                    char       out[static out_cap],
                                                    size_t    *out_len) {
    struct home_ctx *c = (struct home_ctx *) ctx;
    char             req[512];
    if (!agent_json_str(args, "request", sizeof req, req) || req[0] == '\0') {
        return agent_obs(out_cap, out, out_len, "error: missing \"request\"");
    }
    /* One-shot slot: any command that is NOT a confirmation disarms a pending
     * unlock — the confirmation must be the immediately following request. */
    if (!home_is_confirm(req)) {
        home_pending_clear(c);
    }
    /* Collective: "alle Lichter aus" — a group word plus a multi-device match
     * runs the action on every matched WRITABLE device. Locks and refused
     * domains are skipped: a group word never bulk-operates the security
     * boundary. Zero executable devices falls through to the clarify answer. */
    if (home_is_collective(req)) {
        size_t idx[8];
        size_t nm = home_match(c, req, 8, idx);
        if (nm >= 2) {
            static char raw[HOME_OBS_CAP];
            size_t      w = 0, done = 0;
            for (size_t i = 0; i < nm; i++) {
                const struct home_device *d = &c->dev[idx[i]];
                char                      extra[64];
                const char               *service;
                if (!home_domain_writable(d->domain) ||
                    (service = home_action(req, d->domain, sizeof extra, extra)) == nullptr) {
                    continue;
                }
                if (w == 0) {
                    w = (size_t) snprintf(out, out_cap, "OK:");
                }
                if (c->set(c, d, service, extra, sizeof raw, raw) < 0) {
                    w += (size_t) snprintf(
                            out + w, out_cap - w, "%s %s: Fehler", done ? "," : "", d->entity);
                } else {
                    char        num[16];
                    const char *nice =
                            strcmp(service, "turn_on") == 0             ? home_state_word("on")
                            : strcmp(service, "turn_off") == 0          ? home_state_word("off")
                            : strcmp(service, "open_cover") == 0        ? home_state_word("open")
                            : strcmp(service, "close_cover") == 0       ? home_state_word("closed")
                            : home_parse_number(extra, sizeof num, num) ? num
                                                                        : extra;
                    w += (size_t) snprintf(
                            out + w, out_cap - w, "%s %s → %s", done ? "," : "", d->entity, nice);
                    snprintf(c->last_entity, sizeof c->last_entity, "%s", d->entity);
                }
                done++;
            }
            if (done > 0) {
                if (out_len) {
                    *out_len = w;
                }
                return GEIST_OK;
            }
        }
    }
    size_t di = 0;
    if (home_resolve(c, req, out_cap, out, out_len, &di) != 0) {
        return GEIST_OK; /* clarify/error observation already written */
    }
    const struct home_device *d = &c->dev[di];
    if (strcmp(d->domain, "lock") == 0) {
        return home_lock_flow(c, req, d, (long) time(nullptr), out_cap, out, out_len);
    }
    if (!home_domain_writable(d->domain)) {
        return agent_obs(out_cap,
                         out,
                         out_len,
                         "Abgelehnt: '%s' (%s) wird aus Sicherheitsgründen nicht per "
                         "Sprachbefehl gesteuert.",
                         d->entity,
                         d->domain);
    }
    char        extra[64];
    const char *service = home_action(req, d->domain, sizeof extra, extra);
    if (service == nullptr && strcmp(d->domain, "climate") == 0) {
        /* relative setpoint ("mach es wärmer"): read the current value, move
         * it one degree. ponytail: fixed ±1 °C step; make it configurable
         * when a real thermostat asks for finer moves. */
        int dir = home_relative_dir(req);
        if (dir != 0) {
            static char cur_raw[HOME_OBS_CAP];
            char        cur[32];
            if (c->get(c, d, sizeof cur_raw, cur_raw) >= 0 &&
                (ha_json_str(cur_raw, "temperature", sizeof cur, cur) > 0 ||
                 ha_json_str(cur_raw, "state", sizeof cur, cur) > 0)) {
                snprintf(extra, sizeof extra, "\"temperature\":%.4g", strtod(cur, nullptr) + dir);
                service = "set_temperature";
            }
        }
    }
    if (service == nullptr) {
        return agent_obs(out_cap,
                         out,
                         out_len,
                         "Unklar, was mit %s geschehen soll (z.B. ein/aus, Wert angeben).",
                         d->entity);
    }
    static char raw[HOME_OBS_CAP];
    long        rc = c->set(c, d, service, extra, sizeof raw, raw);
    if (rc == -2) {
        return agent_obs(out_cap,
                         out,
                         out_len,
                         "Home Assistant ist nicht konfiguriert (GEIST_HA_URL / GEIST_HA_TOKEN).");
    }
    if (rc < 0) {
        return agent_obs(
                out_cap, out, out_len, "error: Home Assistant nicht erreichbar (%s).", service);
    }
    snprintf(c->last_entity, sizeof c->last_entity, "%s", d->entity); /* pronoun memory */
    char        num[16];
    const char *nice = strcmp(service, "turn_on") == 0             ? home_state_word("on")
                       : strcmp(service, "turn_off") == 0          ? home_state_word("off")
                       : strcmp(service, "open_cover") == 0        ? home_state_word("open")
                       : strcmp(service, "close_cover") == 0       ? home_state_word("closed")
                       : home_parse_number(extra, sizeof num, num) ? num
                                                                   : extra;
    return agent_obs(out_cap, out, out_len, "OK: %s → %s", d->entity, nice);
}

static inline enum geist_status home_status_invoke(void      *ctx,
                                                   size_t     args_len,
                                                   const char args[static args_len],
                                                   size_t     out_cap,
                                                   char       out[static out_cap],
                                                   size_t    *out_len) {
    struct home_ctx *c = (struct home_ctx *) ctx;
    char             req[512];
    if (!agent_json_str(args, "request", sizeof req, req) || req[0] == '\0') {
        return agent_obs(out_cap, out, out_len, "error: missing \"request\"");
    }
    /* Collective read ("sind alle Lichter aus?"): list every matched device. */
    if (home_is_collective(req)) {
        size_t idx[8];
        size_t nm = home_match(c, req, 8, idx);
        if (nm >= 2) {
            static char agg[HOME_OBS_CAP];
            size_t      w = 0;
            for (size_t i = 0; i < nm; i++) {
                const struct home_device *d = &c->dev[idx[i]];
                char                      state[64];
                const char               *word = "Fehler";
                if (c->get(c, d, sizeof agg, agg) >= 0 &&
                    ha_json_str(agg, "state", sizeof state, state) > 0) {
                    word = home_state_word(state);
                }
                w += (size_t) snprintf(
                        out + w, out_cap - w, "%s%s: %s", i ? ", " : "", d->entity, word);
            }
            if (out_len) {
                *out_len = w;
            }
            return GEIST_OK;
        }
    }
    size_t di = 0;
    if (home_resolve(c, req, out_cap, out, out_len, &di) != 0) {
        return GEIST_OK;
    }
    const struct home_device *d = &c->dev[di];
    static char               raw[HOME_OBS_CAP];
    long                      rc = c->get(c, d, sizeof raw, raw);
    if (rc == -2) {
        return agent_obs(out_cap,
                         out,
                         out_len,
                         "Home Assistant ist nicht konfiguriert (GEIST_HA_URL / GEIST_HA_TOKEN).");
    }
    if (rc < 0) {
        return agent_obs(out_cap, out, out_len, "error: Home Assistant nicht erreichbar.");
    }
    snprintf(c->last_entity, sizeof c->last_entity, "%s", d->entity);
    char state[64], unit[16];
    if (ha_json_str(raw, "state", sizeof state, state) == 0) {
        return agent_obs(out_cap, out, out_len, "error: unerwartete Antwort für %s.", d->entity);
    }
    if (ha_json_str(raw, "unit_of_measurement", sizeof unit, unit) > 0) {
        return agent_obs(out_cap, out, out_len, "%s: %s %s", d->entity, state, unit);
    }
    return agent_obs(out_cap, out, out_len, "%s: %s", d->entity, home_state_word(state));
}

/* Ready-made whitelist entries. ctx must be initialized (registry loaded,
 * set/get hooks pointing at home_set_ha/home_get_ha or an eval stub) and must
 * outlive the agent. Names chosen by routing score-probe: distinct first
 * tokens, imperative vs question evidence. */
static inline struct geist_tool home_command_tool(struct home_ctx *c) {
    return (struct geist_tool) {
            .name        = "control_device",
            .description = "ein Hausgerät schalten oder stellen (Licht, Heizung, Rollladen)",
            .args_schema = "{\"request\": string}",
            .invoke      = home_command_invoke,
            .ctx         = c,
    };
}

static inline struct geist_tool home_status_tool(struct home_ctx *c) {
    return (struct geist_tool) {
            .name        = "home_status",
            .description = "den Zustand eines Geräts oder Sensors im Haus abfragen",
            .args_schema = "{\"request\": string}",
            .invoke      = home_status_invoke,
            .ctx         = c,
    };
}

#endif /* GEIST_AGENT_HOME_H */
