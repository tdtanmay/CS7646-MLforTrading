from ManualStrategy import manual_strategy
from StrategyLearner import strategy_learner
from experiment1 import e1
from experiment2 import e2

def author():
    return 'chuang405'

if __name__ == '__main__':
    manual_strategy()
    strategy_learner()
    e1()
    e2()