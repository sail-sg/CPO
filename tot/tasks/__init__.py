def get_task(name):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from tot.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    elif name == 'bamboogle':
        from tot.tasks.bamboogle import FactualQA
        return FactualQA()  
    elif name == '2wiki':
        from tot.tasks.wiki import FactualQA
        return FactualQA()  
    elif name == 'qasc':
        from tot.tasks.qasc import FactualQA
        return FactualQA()  
    elif name == 'fever':
        from tot.tasks.fever import FactualQA
        return FactualQA() 
    else:
        raise NotImplementedError
