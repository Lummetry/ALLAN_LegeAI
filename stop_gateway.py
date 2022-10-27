import psutil

PROCNAME = "python.exe"

if __name__ == "__main__":

    procs = []
    pid = None
    for proc in psutil.process_iter():
        if proc.name() == PROCNAME:
            if proc.cmdline()[1] == "run_gateway.py":
              procs.append(proc)
              break

    for proc in psutil.process_iter():
        if proc.ppid() == procs[0].pid:
            procs.append(proc)

    for proc in procs:
        proc.kill()
