import torch
import torch.nn as nn
from functools import reduce



class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
    
    # ConcatTable : applies each member module to the same input Tensor and outputs a table;
    def forward_prepare(self, input):
        """
        Get the output of all models and gather them in output.
        """
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

    
class Lambda(LambdaBase):
    """
        Apply lambda_func to the output of all models.
    """
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

    
class LambdaMap(LambdaBase):
    """
        Apply lambda_func to each model's output and return the resulted list.
    """
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

    
class LambdaReduce(LambdaBase):
    """
        Apply reduce with lambda_func to the output of each model.
    """
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

    
def block1():
     return nn.Sequential( 
            LambdaMap(lambda x: x, # Concat The results. check forward_prepare function.
                # First model
                nn.Sequential( 
                    nn.Conv2d(16,16,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16,16,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(16),
                ),
                # Second Model
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable Takes a table of tensors and outputs summation of all tensors.
            nn.ReLU(),
        )

    
def block2():
    return nn.Sequential(
            LambdaMap(lambda x: x, # Concat The results.
                # First model
                nn.Sequential( 
                    nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(32),
                ),
                # Second Model
                Lambda(lambda x: x), # Identity,
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable Takes a table of tensors and outputs summation of all tensors.
            nn.ReLU(),
        )

def block3():
    return nn.Sequential(
            LambdaMap(lambda x: x, # Concat The results.
                # First model
                nn.Sequential( # Sequential,
                    nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                    nn.BatchNorm2d(64),
                ),
                # Second Model
                Lambda(lambda x: x), 
            ),
            LambdaReduce(lambda x,y: x+y), # CAddTable Takes a table of tensors and outputs summation of all tensors.
            nn.ReLU(),
        )

def block4():
    return nn.Sequential( 
                LambdaMap(lambda x: x, # Concat The results.
                    # First model
                    nn.Sequential( # Sequential,
                        nn.Conv2d(16,32,(3, 3),(2, 2),(1, 1),1,1,bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                        nn.BatchNorm2d(32),
                    ),
                    # Second Model   
                    nn.Sequential( # Sequential,
                        nn.AvgPool2d((1, 1),(2, 2)),
                        LambdaReduce(lambda x,y,dim=1: torch.cat((x,y),dim), # Concat,
                            Lambda(lambda x: x), # Identity,
                            Lambda(lambda x: x * 0), #resnet mulconstant,
                        ),
                    ),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable Takes a table of tensors and outputs summation of all tensors.
                nn.ReLU(),
            )

def block5():
    return nn.Sequential( 
                LambdaMap(lambda x: x, # Concat The results.
                    # First model
                    nn.Sequential( 
                        nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1),1,1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
                        nn.BatchNorm2d(64),
                    ),
                    # Second Model
                    nn.Sequential(
                        nn.AvgPool2d((1, 1),(2, 2)),
                        LambdaReduce(lambda x,y,dim=1: torch.cat((x,y),dim), # Concat,
                            Lambda(lambda x: x), # Identity,
                            Lambda(lambda x: x * 0), #resnet mulconstant,
                        ),
                    ),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable Takes a table of tensors and outputs summation of all tensors.
                nn.ReLU(),
            )