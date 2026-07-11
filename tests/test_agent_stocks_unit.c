/* test_agent_stocks_unit — the pure (no-network) core of stock_movers:
 * direction detection from the lifted query and screener-JSON formatting
 * (bare-number and {"raw":...} percent variants, API order preserved). */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/agent_stocks.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;

int main(void) {
    /* direction: losers words flip, anything else = gainers */
    fails += geist_expect(!stocks_wants_losers("Welche Aktie hat heute am besten performt?"),
                          "direction: best performer -> gainers");
    fails += geist_expect(stocks_wants_losers("Which stocks lost the most today?"),
                          "direction: lost -> losers");
    fails += geist_expect(stocks_wants_losers("Welche Aktien waren heute die Verlierer?"),
                          "direction: Verlierer -> losers");
    fails += geist_expect(!stocks_wants_losers("show me the closer look"),
                          "direction: 'closer' is not 'loser' (word start)");

    /* formatting: bare-number percent, RE-sorted (the API order is only
     * approximately sorted; rank 1 IS the answer), rank + sign printed */
    const char *bare = "{\"quotes\":[{\"symbol\":\"BBIO\",\"shortName\":\"BridgeBio\","
                       "\"regularMarketChangePercent\":17.068806},"
                       "{\"symbol\":\"LASR\",\"regularMarketChangePercent\":25.63054}]}";
    char        out[512];
    size_t      n = stocks_format(bare, 0, sizeof out, out);
    fails += geist_expect(n > 0 && strstr(out, "1. LASR +25.63%") != nullptr &&
                                  strstr(out, "2. BBIO +17.07%") != nullptr,
                          "format: gainers re-sorted descending, ranked, signed");

    /* formatting: {"raw":...} percent variant; losers sort ascending (worst
     * first) */
    const char *rawv = "{\"quotes\":[{\"symbol\":\"ABC\","
                       "\"regularMarketChangePercent\":{\"raw\":-3.25,\"fmt\":\"-3.25%\"}},"
                       "{\"symbol\":\"XYZ\","
                       "\"regularMarketChangePercent\":{\"raw\":-12.5,\"fmt\":\"-12.5%\"}}]}";
    n                = stocks_format(rawv, 1, sizeof out, out);
    fails += geist_expect(n > 0 && strstr(out, "1. XYZ -12.50%") != nullptr &&
                                  strstr(out, "2. ABC -3.25%") != nullptr,
                          "format: losers re-sorted ascending, raw-object percent");

    /* formatting: garbage in -> 0 out (invoke turns that into an error obs) */
    fails += geist_expect(stocks_format("<html>rate limited</html>", 0, sizeof out, out) == 0,
                          "format: non-JSON yields nothing");

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("stock_movers: direction + screener formatting pass\n");
    return GEIST_TEST_PASS;
}
