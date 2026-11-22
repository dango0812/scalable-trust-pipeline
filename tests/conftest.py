from hypothesis import settings

# Property tests run the same examples on every machine and in CI, and leave no example
# database behind in the working tree.
settings.register_profile("repo", derandomize=True, database=None, deadline=None)
settings.load_profile("repo")
