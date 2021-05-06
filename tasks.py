from invoke import task

@task
def run(ctx, k = 0, n = 0, Random=True):
    ctx.run("python3.8.5 setup.py build_ext --inplace")
    try:
        if Random:
            ctx.run(f"python3.8.5 main.py {k} {n}")
        else:
            if not k or not n:
                raise Exception('Some required arguments are missing')
            ctx.run(f"python3.8.5 main.py {k} {n} --no-Random")
    finally: #running succeeded or not - cleaning the shared object built
        ctx.run("rm mykmeanssp.cpython-38-x86_64-linux-gnu.so")