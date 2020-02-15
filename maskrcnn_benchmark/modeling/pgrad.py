# SSY
import torch.autograd
from torch.autograd import Function
from torch.nn import Module 
# SSY
#def float2bf16_(xmlp):
#        ssign=torch.sign(xmlp)
#        aabs=torch.abs(xmlp)
#        a=torch.where(aabs==0.0,aabs,torch.log2(aabs))
#        b_exp=torch.floor(a) # exp value
#        rnd2exp=torch.pow(2,b_exp) # get back the old value rounded to 2 exp
#        mantis=torch.div(aabs,rnd2exp) # mantis
#        m256=torch.mul(mantis,256)
#        f256=torch.floor(m256)
#        d256=torch.div(f256,256)
#        res=torch.mul(d256,rnd2exp)
#        res1= torch.mul(res,ssign)
#        # already debug to remove nan from log(0)
#        #print("xmlp "+str(xmlp))
#        #print("res1 "+str(res1))
#        return res1

# SSY avoid lots of intermedia result space
def float2bf16(xmlp):
        aabs=torch.abs(xmlp)
        rnd2exp=torch.pow(2,torch.floor(torch.where(aabs==0.0,aabs,torch.log2(aabs)))) # get back the old value rounded to 2 exp
        return  torch.mul(torch.mul(torch.div(torch.floor(torch.mul(torch.div(aabs,rnd2exp),4)),4),rnd2exp),torch.sign(xmlp))

class bf16cutfp(Function):
    @staticmethod
    def forward(ctx, tensor):
        return float2bf16(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class bf16cutbp(Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return float2bf16(grad_output)

class bf16cutfp_mod(Module):
  def __init__(self):
    super(bf16cutfp_mod,self).__init__()

  def forward(self,input):
    return bf16cutfp.apply(input)

class bf16cutbp_mod(Module):
  def __init__(self):
    super(bf16cutbp_mod,self).__init__()

  def forward(self,input):
    return bf16cutbp.apply(input)
