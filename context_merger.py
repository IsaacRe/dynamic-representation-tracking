class ContextMerger:
    """
    Simple class that merges multiple context objects
    """
    def __init__(self, *contexts):
        self.contexts = contexts

    def __enter__(self):
        for c in self.contexts:
            c.__enter__()

    def __exit__(self, *exc_vars):
        for c in self.contexts:
            c.__exit__(*exc_vars)
