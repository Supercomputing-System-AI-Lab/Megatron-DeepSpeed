// ftv2_dhang.c (ft-sys-v2 heartbeat tests, 2026-11-06): the calling thread enters state D, the way a thread blocked inside a
// driver call shows, so that a blocking stack dump (py-spy's ptrace attach) cannot stop it (V3-2L 5863006, frontier02408).
// ftv2_vfork_hang(): vfork(); the child asks for SIGKILL when its parent thread dies, then pauses for ever (it never execs or
// exits, so the parent's vfork never completes: the parent thread waits in wait_for_vfork_done, state D, TASK_KILLABLE on this
// kernel, so SIGKILL still ends it, and the child follows by PR_SET_PDEATHSIG). Only raw system calls run in the child.
// Build: gcc -O2 -shared -fPIC -o ftv2_dhang.so ftv2_dhang.c
#define _GNU_SOURCE
#include <signal.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

int ftv2_vfork_hang(void)
{
    pid_t c = vfork();
    if (c == 0) {
        syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
        for (;;)
            syscall(SYS_pause);
    }
    return (int)c;            /* reached only if the child ends */
}
