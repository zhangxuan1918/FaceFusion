def get_strategy_scope(strategy):
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = DummyContextManager()

    return strategy_scope


class DummyContextManager(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
