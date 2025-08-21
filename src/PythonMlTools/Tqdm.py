def get_tqdm():
    try:
        # Detecta se estamos em um notebook
        from IPython import get_ipython
        if "IPKernelApp" in get_ipython().config:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except Exception:
        # Se não estiver em Jupyter, usa o tqdm normal
        from tqdm import tqdm
    return tqdm
