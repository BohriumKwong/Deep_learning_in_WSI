# @Time    : 2019.03.25
# @Author  : Bohrium Kwong
# @Licence : bio-totem

import sys
sys.path.append('../')
from normalization.macenko import MacenkoNormalizer
from normalization.vahadane import VahadaneNormalizer
#from utils import misc_utils as mu


def standard_transfrom(standard_img,method = 'M'):
    if method == 'V':
        stain_method = VahadaneNormalizer()
        stain_method.fit(standard_img)
    else: 
        stain_method = MacenkoNormalizer()
        stain_method.fit(standard_img)
    return stain_method
