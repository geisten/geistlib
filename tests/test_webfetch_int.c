/*
 * test_webfetch_int — the real web_fetch path (fork+execvp curl -> read ->
 * strip) against a loopback HTTP server the test spawns itself. Deterministic,
 * no external network. Also checks the security rejections through invoke().
 * SKIPs if curl is not installed. No assert() — exit code carries PASS/FAIL.
 */
#ifndef _GNU_SOURCE /* socket extras (INADDR_LOOPBACK …); guard so a build that */
#define _GNU_SOURCE /* already passes -D_GNU_SOURCE (the musl CI job) doesn't redefine */
#endif

#include "test_helpers.h"

#include "../tools/agent_webfetch.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails = 0;
/* fork+exec `curl --version`; return 1 if curl is present and runnable. */
static int curl_present(void) {
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    }
    if (pid == 0) {
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) {
            dup2(nul, STDOUT_FILENO);
            dup2(nul, STDERR_FILENO);
        }
        char *const argv[] = {(char *) "curl", (char *) "--version", nullptr};
        execvp("curl", argv);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

/* Bind a loopback listener on an ephemeral port; return fd and *port, or -1. */
static int serve_listen(int *port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family         = AF_INET;
    a.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
    a.sin_port           = 0; /* ephemeral */
    if (bind(fd, (struct sockaddr *) &a, sizeof a) != 0 || listen(fd, 1) != 0) {
        close(fd);
        return -1;
    }
    socklen_t al = sizeof a;
    if (getsockname(fd, (struct sockaddr *) &a, &al) != 0) {
        close(fd);
        return -1;
    }
    *port = ntohs(a.sin_port);
    return fd;
}

#define BODY \
    "<html><body><h1>Hi</h1><p>warranty 24 months</p><script>var x=1;</script></body></html>"

int main(void) {
    if (!curl_present()) {
        GEIST_SKIP("curl not installed");
    }

    char   out[8192];
    size_t n = 0;

    /* ---- security rejections (no network needed) ---- */
    webfetch_invoke((void *) (intptr_t) "example.com",
                    strlen("{\"url\":\"file:///etc/passwd\"}"),
                    "{\"url\":\"file:///etc/passwd\"}",
                    sizeof out,
                    out,
                    &n);
    fails += geist_expect(strstr(out, "only http/https") != nullptr, "reject: file:// scheme");

    webfetch_invoke((void *) (intptr_t) "example.com",
                    strlen("{\"url\":\"http://127.0.0.1/x\"}"),
                    "{\"url\":\"http://127.0.0.1/x\"}",
                    sizeof out,
                    out,
                    &n);
    fails += geist_expect(strstr(out, "not allowed") != nullptr, "reject: host off the allowlist");

    webfetch_invoke(nullptr, strlen("{\"q\":\"x\"}"), "{\"q\":\"x\"}", sizeof out, out, &n);
    fails += geist_expect(strstr(out, "missing") != nullptr, "reject: missing url");

    /* ---- real fetch from a loopback server ---- */
    int port = 0;
    int lfd  = serve_listen(&port);
    if (lfd < 0) {
        GEIST_SKIP("could not bind a loopback socket");
    }
    pid_t srv = fork();
    if (srv < 0) {
        close(lfd);
        GEIST_SKIP("fork failed");
    }
    if (srv == 0) { /* one-shot server */
        int c = accept(lfd, nullptr, nullptr);
        if (c >= 0) {
            char req[1024];
            (void) !read(c, req, sizeof req); /* drain the request line (gcc: must "use" it) */
            const char *resp = "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\n\r\n" BODY;
            (void) !write(c, resp, strlen(resp));
            close(c);
        }
        close(lfd);
        _exit(0);
    }
    close(lfd); /* parent doesn't accept */

    char url[64];
    snprintf(url, sizeof url, "{\"url\":\"http://127.0.0.1:%d/\"}", port);
    enum geist_status st =
            webfetch_invoke((void *) (intptr_t) "127.0.0.1", strlen(url), url, sizeof out, out, &n);
    int wst = 0;
    waitpid(srv, &wst, 0);

    fprintf(stderr, "fetch -> st=%d, %zu bytes: \"%.120s\"\n", (int) st, n, out);
    fails += geist_expect(st == GEIST_OK, "fetch: returns OK");
    fails +=
            geist_expect(strstr(out, "warranty 24 months") != nullptr, "fetch: body text returned");
    fails += geist_expect(strchr(out, '<') == nullptr, "fetch: HTML tags stripped");

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("web_fetch: live loopback fetch + rejections pass\n");
    return GEIST_TEST_PASS;
}
